import numpy as np
import torch
import torch.nn.functional as F
import cv2
from numba import njit, prange


def pred_sub_pix(img, kpts, ids, region=(8, 8)):
    return corner_sub_pix(img, kpts, region=region)


def corner_sub_pix(img, corners, region=(8, 8)):
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    return cv2.cornerSubPix(img, np.expand_dims(corners,
                                                axis=1).astype(np.float32),
                            region, (-1, -1), term).squeeze(1)


def extract_patches(img: np.ndarray, keypoints: np.ndarray, patch_size: int = 24) -> np.ndarray:
    padding = patch_size // 2

    # Pad the image with zeros
    padded_img = F.pad(img.squeeze(0), (padding, padding, padding, padding), mode='constant', value=0)

    # Extract the patches centered around the keypoints
    patches = [padded_img[kp[1]:kp[1] + 2 * padding, kp[0]:kp[0] + 2 * padding]
               for kp in keypoints]
    return torch.stack(patches)

# @torch.jit.script  # TODO
def speedy_bargmax2d(x):
    _, indices = torch.max(x.view(x.shape[0], -1), dim=1)
    col_indices = indices % x.shape[2]
    row_indices = indices // x.shape[2]
    return torch.stack((col_indices, row_indices), dim=1)


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
        loc_hat = torch.expand_dims(loc_hat, axis=0)
        ids_hat = torch.expand_dims(ids_hat, axis=0)

    # (N, C, H/8, W/8)
    ids_argmax = torch.argmax(ids_hat, axis=1)
    loc_argmax = torch.argmax(loc_hat, axis=1)

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
    roi = torch.argwhere(mask)
    ids_found = ids[mask]
    region_pixel = loc[mask]

    # Recover exact pixel in original resolution
    xs = 8 * roi[:, -1] + (region_pixel % 8)
    ys = 8 * roi[:, -2] + (region_pixel / 8).to(torch.int)
    return torch.cat((xs.unsqueeze(1), ys.unsqueeze(1)), dim=1), ids_found