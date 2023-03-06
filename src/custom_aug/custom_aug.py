import random
import numpy as np
import cv2
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.utils import (
    get_opencv_dtype_from_numpy,
    preserve_shape,
)
import custom_aug.utils as utils


class PasteBoard(DualTransform):
    """
    Paste board image on top of arbitrary image
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        target = params['target']
        mask = params['mask'].astype(bool)
        isnegative = params['isnegative']

        if isnegative:
            return target
        assert img.shape == target.shape
        target[mask] = img[mask]
        return target

    @property
    def targets_as_params(self) -> list[str]:
        return ['target', 'mask', 'isnegative']

    def apply_to_mask(self, img, **params):
        return img

    def get_params_dependent_on_targets(self, params):
        return params

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint
    
    def apply_to_keypoints(self, keypoints, **params):
        if params['isnegative']:
            return np.full_like(keypoints, fill_value=-1)
        return keypoints

    def get_transform_init_args_names(self):
        return None


class HistogramMatching(ImageOnlyTransform):
    """
    Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches
    the histogram of the reference image. If the images have multiple channels, the matching is done independently
    for each channel, as long as the number of channels is equal in the input image and the reference.
    Histogram matching can be used as a lightweight normalisation for image processing,
    such as feature matching, especially in circumstances where the images have been taken from different
    sources or in different conditions (i.e. lighting).
    See:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html
    Args:
        reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
            or list of reference images.
        blend_ratio (float, float): Tuple of min and max blend ratio. Matched image will be blended with original
            with random blend factor for increased diversity of generated images.
        read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
            array of image pixels.
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image
    Image types:
        uint8, uint16, float32
    """

    def __init__(
        self,
        blend_ratio=(0.5, 1.0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.blend_ratio = blend_ratio

    def apply(self, img, blend_ratio=0.5, **params):
        reference_image = params['target']
        mask = params['mask']
        return apply_histogram(img, reference_image, mask, blend_ratio)

    def get_params(self):
        return {
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_params_dependent_on_targets(self, params):
        return params

    def get_transform_init_args_names(self):
        return ("blend_ratio")

    @property
    def targets_as_params(self) -> list[str]:
        return ['target', 'mask']

    def _to_dict(self):
        raise NotImplementedError("HistogramMatching can not be serialized.")


@preserve_shape
def apply_histogram(img: np.ndarray, reference_image: np.ndarray, mask, blend_ratio: float) -> np.ndarray:
    if img.dtype != reference_image.dtype:
        raise RuntimeError(
            f"Dtype of image and reference image must be the same. Got {img.dtype} and {reference_image.dtype}"
        )
    assert img.shape == reference_image.shape  # This will assert dims. This should never run
    img, reference_image = np.squeeze(img), np.squeeze(reference_image)

    try:
        matched = match_histograms(img, reference_image, mask, channel_axis=2 if len(img.shape) == 3 else None)
    except TypeError:
        matched = match_histograms(img, reference_image, mask, multichannel=True)  # case for scikit-image<0.19.1
    img = cv2.addWeighted(
        matched,
        blend_ratio,
        img,
        1 - blend_ratio,
        0,
        dtype=get_opencv_dtype_from_numpy(img.dtype),
    )
    return img


@utils.channel_as_last_axis(channel_arg_positions=(0, 1))
@utils.deprecate_multichannel_kwarg()
def match_histograms(image, reference, mask, *, channel_axis=None,
                     multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    multichannel : bool, optional
        Apply the matching separately for each channel. This argument is
        deprecated: specify `channel_axis` instead.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            mask = mask.astype(bool)
            matched_channel = _match_cumulative_cdf(image[mask, channel],
                                                    reference[..., channel])
            matched[mask, channel] = matched_channel
    else:
        raise NotImplementedError
        # _match_cumulative_cdf will always return float64 due to np.interp
        # matched = _match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = utils._supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)
