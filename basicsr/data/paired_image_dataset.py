import numpy as np
import tifffile
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Supported modes:
    1. lmdb
    2. meta_info_file
    3. folder

    ProjectionDataset adaptation:
    ----------------------------
    Add a new task: ``projection_raw``

    For ``projection_raw``:
    - read LQ / GT directly from paired tif files
    - preserve original float32 projection values
    - do NOT divide by 255
    - keep single-channel grayscale structure
    - use paired crop / augment only in training
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt']
            )
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.opt['meta_info_file'],
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.filename_tmpl,
                self.task,
            )

    def _read_projection_tif(self, path):
        img = tifffile.imread(path)

        # keep raw values; do not normalize by 255
        img = np.asarray(img).astype(np.float32)

        # enforce HWC
        if img.ndim == 2:
            img = img[..., None]
        elif img.ndim == 3:
            # if tif accidentally stores shape (1, H, W) or similar
            if img.shape[0] == 1 and img.shape[2] != 1:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] != 1 and img.shape[0] != 1 and img.shape[1] != 1:
                # keep first channel only if an unexpected multi-channel tif appears
                img = img[..., :1]
        else:
            raise ValueError(f'Unsupported tif ndim={img.ndim} for path: {path}')

        return img

    def __getitem__(self, index):
        scale = self.opt['scale']

        # ---------------------------------------------------------
        # projection_raw: direct paired raw projection reading
        # ---------------------------------------------------------
        if self.task == 'projection_raw':
            gt_path = self.paths[index]['gt_path']
            lq_path = self.paths[index]['lq_path']

            img_gt = self._read_projection_tif(gt_path)
            img_lq = self._read_projection_tif(lq_path)

            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                img_gt, img_lq = augment(
                    [img_gt, img_lq],
                    self.opt['use_hflip'],
                    self.opt['use_rot']
                )

            if self.opt['phase'] != 'train':
                img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

            # grayscale / single-channel raw projection, so no BGR<->RGB conversion
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)

            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

        # ---------------------------------------------------------
        # original code paths
        # ---------------------------------------------------------
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if self.task == 'CAR':
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)

            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        elif self.task == 'denoising_gray':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)

            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)

            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)

            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)

        else:
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt['use_hflip'],
                self.opt['use_rot']
            )

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
