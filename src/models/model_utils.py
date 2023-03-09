import numpy as np
import cv2
from numba import njit, prange


def pred_sub_pix(img, kpts, ids, region=(8, 8)):
    return corner_sub_pix(img, kpts, region=region)


def corner_sub_pix(img, corners, region=(8, 8)):
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    return cv2.cornerSubPix(img, np.expand_dims(corners,
                                                axis=1).astype(np.float32),
                            region, (-1, -1), term).squeeze(1)


def extract_patches(img: np.ndarray, keypoints: np.ndarray, patch_size: int = 24) -> list:
    # Compute the half size of the patch
    half_size = patch_size // 2

    # Compute the amount of padding needed in each direction
    padding = half_size

    # Pad the image with zeros
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Extract the patches centered around the keypoints
    patches = []
    for kp in keypoints:
        patch_top = kp[1] + padding - half_size
        patch_bottom = kp[1] + padding + half_size
        patch_left = kp[0] + padding - half_size
        patch_right = kp[0] + padding + half_size
        patch = padded_img[patch_top:patch_bottom, patch_left:patch_right]
        patches.append(patch)
    return patches


@njit('i8[:,::1](f4[:,:,::1])', cache=True, parallel=True)
def speedy_bargmax2d(x):
    max_indices = np.zeros((x.shape[0], 2), dtype=np.int64)
    for i in prange(x.shape[0]):
        maxTemp = np.argmax(x[i])
        max_indices[i] = [maxTemp % x.shape[2], maxTemp // x.shape[2]]
    return max_indices


def pre_bgr_image(image, is_gray=False):
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[..., np.newaxis].astype(np.float32)
    image = (image - 128) / 255  # Well we started with this one so...
    image = image.transpose((2, 0, 1))
    return image


def pred_argmax(loc_hat: np.ndarray, ids_hat: np.ndarray, dust_bin_ids: int):
    """
    Convert a model prediction to label format having class indices at each position.
    Use label_to_keypoints to convert the returned label to keypoints

    Parameters
    ----------
    loc_hat: np.ndarray
        localization output of the model
    ids_hat: np.ndarray
        identities output of the model
    dust_bin_ids: int
        the null id of identities
    """
    if loc_hat.ndim == 3:
        loc_hat = np.expand_dims(loc_hat, axis=0)
        ids_hat = np.expand_dims(ids_hat, axis=0)

    # (N, C, H/8, W/8)
    ids_argmax = np.argmax(ids_hat, axis=1)
    loc_argmax = np.argmax(loc_hat, axis=1)

    # Mask ids_hat using loc_hat dust_bin
    # This way we will do an argmax only over best ids with valid location
    ids_argmax[loc_argmax == 64] = dust_bin_ids
    return loc_argmax, ids_argmax


def pred_to_keypoints(loc_hat: np.ndarray, ids_hat: np.ndarray, dust_bin_ids: int):
    """
    Transform a model prediction to keypoints with ids and optionally confidences
    """
    assert loc_hat.ndim == 4 and ids_hat.ndim == 4
    loc_argmax, ids_argmax = pred_argmax(loc_hat, ids_hat, dust_bin_ids)
    kpts, ids = label_to_keypoints(loc_argmax, ids_argmax, dust_bin_ids)
    return kpts, ids


def label_to_keypoints(loc: np.ndarray, ids: np.ndarray, dust_bin_ids: int):
    """
    Convert a label like format with class indices to keypoints in original resolution

    Parameters
    ----------
    loc
        localization map for corners
    ids
        identities map for corners
    dust_bin_ids
        the null id of identities
    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        array of keypoints and associated ids
    """
    assert loc.ndim == 3 and ids.ndim == 3

    # Find in which regions the corners are
    mask = ids != dust_bin_ids
    roi = np.argwhere(mask)
    ids_found = ids[mask]
    region_pixel = loc[mask]

    # Recover exact pixel in original resolution
    xs = 8 * roi[:, -1] + (region_pixel % 8)
    ys = 8 * roi[:, -2] + (region_pixel / 8).astype(int)
    return np.c_[xs, ys], ids_found
