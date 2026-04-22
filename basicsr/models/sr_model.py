import os
from collections import OrderedDict
from os import path as osp

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


def tensor_to_raw_image(tensor):
    """Convert a tensor to raw numpy array without normalization.

    Input:
        (1, C, H, W) or (C, H, W)

    Output:
        - grayscale: (H, W)
        - color: (H, W, C)
    """
    if isinstance(tensor, list):
        tensor = tensor[0]

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    arr = tensor.detach().float().cpu().numpy()

    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = np.squeeze(arr, axis=2)

    return arr.astype(np.float32)


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for image restoration, adapted for ProjectionDataset."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt['path'].get('strict_load_g', True),
                    'params_ema'
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                raise ValueError(op)
            return torch.Tensor(tfnp).to(self.device)

        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')

        output = torch.cat(out_list, dim=0)
        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        metrics_on_raw = self.opt['val'].get('metrics_on_raw', False)
        save_img_as_tif = self.opt['val'].get('save_img_as_tif', False)
        keep_original_name = self.opt['val'].get('keep_original_name', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                self._initialize_best_metric_results(dataset_name)

            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            if metrics_on_raw or save_img_as_tif:
                sr_img_raw = tensor_to_raw_image(visuals['result'])
                metric_data['img'] = sr_img_raw
                if 'gt' in visuals:
                    gt_img_raw = tensor_to_raw_image(visuals['gt'])
                    metric_data['img2'] = gt_img_raw
            else:
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    metric_data['img2'] = gt_img

            if 'gt' in visuals:
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    base_dir = osp.join(self.opt['path']['visualization'], img_name)
                    if save_img_as_tif:
                        save_img_path = osp.join(base_dir, f'{img_name}_{current_iter}.tif')
                        tifffile.imwrite(save_img_path, metric_data['img'].astype(np.float32))
                    else:
                        save_img_path = osp.join(base_dir, f'{img_name}_{current_iter}.png')
                        imwrite(metric_data['img'], save_img_path)
                else:
                    base_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    os.makedirs(base_dir, exist_ok=True)

                    if save_img_as_tif:
                        if keep_original_name:
                            save_name = f'{img_name}.tif'
                        elif self.opt['val'].get('suffix'):
                            save_name = f'{img_name}_{self.opt["val"]["suffix"]}.tif'
                        else:
                            save_name = f'{img_name}_{self.opt["name"]}.tif'
                        save_img_path = osp.join(base_dir, save_name)
                        tifffile.imwrite(save_img_path, metric_data['img'].astype(np.float32))
                    else:
                        if keep_original_name:
                            save_name = f'{img_name}.png'
                        elif self.opt['val'].get('suffix'):
                            save_name = f'{img_name}_{self.opt["val"]["suffix"]}.png'
                        else:
                            save_name = f'{img_name}_{self.opt["name"]}.png'
                        save_img_path = osp.join(base_dir, save_name)
                        imwrite(metric_data['img'], save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.6f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.6f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g',
                current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
