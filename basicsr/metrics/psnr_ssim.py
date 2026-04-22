import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


def _resolve_data_range(img, img2, data_range):
    if data_range is not None:
        return float(data_range)

    max_val = max(float(np.max(img)), float(np.max(img2)))
    min_val = min(float(np.min(img)), float(np.min(img2)))
    dynamic_range = max_val - min_val

    if dynamic_range <= 0:
        dynamic_range = 1.0
    return dynamic_range


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, data_range=None, **kwargs):
    """Calculate PSNR.

    Modification:
    - support arbitrary float raw projection values
    - use ``data_range`` if given
    - otherwise infer a dynamic range from the pair
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')

    dr = _resolve_data_range(img, img2, data_range)
    return 20.0 * np.log10(dr / np.sqrt(mse))


def _ssim(img, img2, data_range):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, data_range=None, **kwargs):
    """Calculate SSIM for arbitrary float projection values."""
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    dr = _resolve_data_range(img, img2, data_range)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i], dr))
    return np.array(ssims).mean()
