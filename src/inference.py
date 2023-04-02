import cv2
from dataclasses import replace
import numpy as np
import torch

from aruco_utils import draw_inner_corners, label_to_keypoints, cv2_aruco_detect
from typing import Optional
import configs
from configs import load_configuration
from models.model_utils import pred_to_keypoints, extract_patches, pre_bgr_image
from models.net import lModel, dcModel
from models.refinenet import RefineNet, lRefineNet


def solve_pnp(keypoints, col_count, row_count, square_len, camera_matrix, dist_coeffs):
    if keypoints.shape[0] < 4:
        return False, None, None

    # Create inner corners board points
    inn_rc = np.arange(1, row_count)
    inn_cc = np.arange(1, col_count)
    object_points = np.zeros(((col_count - 1) * (row_count - 1), 3), np.float32)
    object_points[:, :2] = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * square_len

    image_points = keypoints[:, :2].astype(np.float32)
    object_points_found = object_points[keypoints[:, 2].astype(int)]

    ret, rvec, tvec = cv2.solvePnP(object_points_found, image_points, camera_matrix, dist_coeffs)
    return ret, rvec, tvec


def infer_image(img: np.ndarray, dust_bin_ids: int, deepc: lModel,
                refinenet: Optional[lRefineNet] = None,
                draw_pred: bool = False):
    """
    Do full inference on a BGR image
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = pre_bgr_image(img_gray, is_gray=True)
    loc_hat, ids_hat = deepc.infer_image(img_gray, preprocessing=False)
    kpts_hat, ids_found = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)

    # Draw predictions in RED
    if draw_pred:
        img = draw_inner_corners(img, kpts_hat, ids_found, radius=3,
                                 draw_ids=True, color=(0, 0, 255))

    if ids_found.shape[0] == 0:
        return np.array([]), img

    if refinenet is not None:
        patches = extract_patches(img_gray, kpts_hat)

        # Extract 8x refined corners (in original resolution)
        refined_kpts, _ = refinenet.infer_patches(patches, kpts_hat)

        # Draw refinenet refined corners in yellow
        if draw_pred:
            img = draw_inner_corners(img, refined_kpts, ids_found,
                                     draw_ids=False, radius=1, color=(0, 255, 255))

    keypoints = refined_kpts if refinenet else kpts_hat
    keypoints = np.array([[k[0], k[1], idx] for k, idx in sorted(zip(keypoints,
                                                                     ids_found),
                                                                 key=lambda x:
                                                                 x[1])])
    return keypoints, img


def load_models(deepc_ckpt: str, refinenet_ckpt: Optional[str] = None, n_ids: int = 16, device='cuda'):
    deepc = lModel.load_from_checkpoint(deepc_ckpt, dcModel=dcModel(n_ids))
    deepc.eval()
    deepc.to(device)

    refinenet = None
    if refinenet_ckpt is not None:
        refinenet = lRefineNet.load_from_checkpoint(refinenet_ckpt, refinenet=RefineNet())
        refinenet.eval()
        refinenet.to(device)

    return deepc, refinenet


if __name__ == '__main__':
    import os
    from gridwindow import MagicGrid
    from utils import pixel_error
    from data import CharucoDataset
    from aruco_utils import get_aruco_dict, get_board
    config = load_configuration(configs.CONFIG_PATH)

    # Load aruco board for cv2 inference
    dictionary = get_aruco_dict(config.board_name)
    board = get_board(config)
    parameters = cv2.aruco.DetectorParameters_create()

    # Load models
    deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
    refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids=config.n_ids, device=device)

    # Inference test on validation data
    up_scale = 1  # Set to 8 for with/without refinenet comparison
    if up_scale > 1:
        config = replace(config, input_size=(320 * up_scale, 240 * up_scale))

    # Load val dataset
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    if "DISPLAY" in os.environ:
        w = MagicGrid(1200, 1200, waitKey=0)
    d_tot = 0
    d_ref_tot = 0
    for ith, sample in enumerate(dataset_val):
        image, label = sample.values()
        loc, ids = label

        # Images returned from dataset are normalized.
        img = ((image * 255) + 128).astype(np.uint8)
        img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

        if up_scale > 1:
            img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR)

        # Run inference
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet, draw_pred=True)
        print('Keypoints\n', keypoints)

        if up_scale > 1:
            # Run inference again without refinenet
            keypoints_raw, _ = infer_image(img, config.n_ids, deepc, None,
                                           draw_pred=False)

            label_kpts, label_ids = label_to_keypoints(loc[None, ...], ids[None, ...], config.n_ids)
            label_kpts = label_kpts.astype(np.float32) / up_scale

            label_kpts = np.array([[k[0], k[1], idx] for k, idx in
                                  sorted(zip(label_kpts, label_ids), key=lambda x:
                                         x[1])])

            if len(label_kpts) != 0 and len(keypoints) != 0:
                d, d_ref = pixel_error(keypoints_raw, keypoints, label_kpts)

            # Error statistics
            if d is not None:
                d_tot += d
                d_ref_tot += d_ref

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters)

        # Statistics up now
        if up_scale > 1:
            print('Cumulative statistics on samples (up now)')
            print(f'Mean Error raw: {d_tot / (ith + 1):.2f}')
            print(f'Mean Error ref: {d_ref_tot / (ith + 1):.2f}')

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if "DISPLAY" in os.environ:
            if w.update([out_img_dc, out_img_cv]) == ord('q'):
                break

    # Inference test on custom image
    SAMPLE_IMAGES = './reference/samples_test/IMG_7412.png'
    import glob
    for p in glob.glob(SAMPLE_IMAGES):
        img = cv2.imread(p)

        # Run inference
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet,
                                            draw_pred=True)
        print(keypoints)

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters)

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if "DISPLAY" in os.environ:
            if w.update([out_img_dc, out_img_cv]) == ord('q'):
                break
