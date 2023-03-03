import numpy as np
import cv2
from numba import njit, prange


def pred_sub_pix(img, loc, ids, dust_bin_ids, region=(8, 8)):
    kps, ids = label_to_keypoints(loc, ids, dust_bin_ids)
    if not kps.shape[0]:
        return kps, ids
    return corner_sub_pix(img, kps, region=region), ids


def corner_sub_pix(img, corners, region=(8, 8)):
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    return cv2.cornerSubPix(img, np.expand_dims(corners,
                                                axis=1).astype(np.float32),
                            region, (-1, -1), term).squeeze(1)


def extract_patches(img: np.ndarray, keypoints: np.ndarray, patch_size: int = 24) -> tuple:
    # Compute the half size of the patch
    half_size = patch_size // 2
    
    # Compute the borders to pad the image with
    top, bottom = half_size, half_size
    left, right = half_size, half_size
    
    if keypoints[:, 0].min() < half_size:
        left = half_size - keypoints[:, 0].min()
    if keypoints[:, 1].min() < half_size:
        top = half_size - keypoints[:, 1].min()
    if keypoints[:, 0].max() >= img.shape[1] - half_size:
        right = half_size + keypoints[:, 0].max() - (img.shape[1] - 1)
    if keypoints[:, 1].max() >= img.shape[0] - half_size:
        bottom = half_size + keypoints[:, 1].max() - (img.shape[0] - 1)
    
    # Pad the image with zeros
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Extract the patches centered around the keypoints
    patches = []
    paddings = []
    for kp in keypoints:
        patch_top = kp[1] - half_size + top
        patch_bottom = kp[1] + half_size + bottom
        patch_left = kp[0] - half_size + left
        patch_right = kp[0] + half_size + right

        left_p = max(0, half_size - kp[0])
        top_p = max(0, half_size - kp[1])
        
        patch = padded_img[patch_top:patch_bottom, patch_left:patch_right]
        patches.append(patch)
        paddings.append([left_p, top_p])
    
    return patches, np.array(paddings)


@njit('i8[:,::1](f4[:,:,::1])', cache=True, parallel=True)
def speedy_bargmax2d(x):
    max_indices = np.zeros((x.shape[0],2),dtype=np.int64)
    for i in prange(x.shape[0]):
        maxTemp = np.argmax(x[i])
        max_indices[i] = [maxTemp // x.shape[2], maxTemp % x.shape[2]]
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
    loc_hat
        localization output of the model
    ids_hat
        identities output of the model
    dust_bin_ids
        the null id of identities
    """
    if loc_hat.ndim == 3:
        loc_hat = np.expand_dims(loc_hat, axis=0)
        ids_hat = np.expand_dims(ids_hat, axis=0)

    # (N, C, H/8, W/8) -> (N, H/8, W/8, C)
    loc_hat = np.transpose(loc_hat, (0, 2, 3, 1))
    ids_hat = np.transpose(ids_hat, (0, 2, 3, 1))

    ids_argmax = np.argmax(ids_hat, axis=-1)
    loc_argmax = np.argmax(loc_hat, axis=-1)
    # TODO: Add debug informations for other ids prob on same position etc
    # also loc infos for multiple corners hypothesis in same region

    return loc_argmax, ids_argmax


def label_to_keypoints(loc: np.ndarray, ids: np.ndarray,
                       dust_bin_ids: int) -> list[tuple[tuple, int]]:
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
    try:
        loc = loc.squeeze(0)
        ids = ids.squeeze(0)
    except ValueError:
        pass

    assert loc.ndim == 2 and ids.ndim == 2

    # Find in which regions the corners are
    roi = np.argwhere(ids != dust_bin_ids)
    ids_found = ids[ids != dust_bin_ids]

    region_pixel = loc[ids != dust_bin_ids]

    # Recover exact pixel in original resolution
    xs = 8 * roi[:, 1] + (region_pixel % 8)
    ys = 8 * roi[:, 0] + (region_pixel / 8).astype(int)
    return np.c_[xs, ys], ids_found
