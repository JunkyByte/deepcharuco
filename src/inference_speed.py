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


@profile
def infer_image(img: np.ndarray, dust_bin_ids: int, deepc: lModel,
                refinenet: Optional[lRefineNet] = None,
                draw_pred: bool = False,
                device='cpu'):
    """
    Do full inference on a BGR image
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = pre_bgr_image(img_gray)
    img_gray = torch.tensor(img_gray, device=device)  # TODO check me
    loc_hat, ids_hat = deepc.infer_image(img_gray, preprocessing=False)
    keypoints, ids_found = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)

    # Draw predictions in RED
    if draw_pred:
        img = draw_inner_corners(img, keypoints.cpu().numpy(), ids_found.cpu().numpy(), radius=3,
                                 draw_ids=True, color=(0, 0, 255))

    if ids_found.shape[0] == 0:
        return np.array([]), img

    if refinenet is not None:
        patches = extract_patches(img_gray, keypoints)
        # Extract 8x refined corners (in original resolution)
        keypoints, _ = refinenet.infer_patches(patches, keypoints)

    keypoints = keypoints.cpu().numpy()
    ids_found = ids_found.cpu().numpy()

    if draw_pred and refinenet is not None:
        # Draw refinenet refined corners in yellow
        if draw_pred:
            img = draw_inner_corners(img, keypoints, ids_found,
                                     draw_ids=False, radius=1, color=(0, 255, 255))

    keypoints = np.array([[k[0], k[1], idx] for k, idx in sorted(zip(keypoints, ids_found),
                                                                 key=lambda x: x[1])])
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

def inf_no_prof(device):
    import os
    from gridwindow import MagicGrid
    from utils import pixel_error
    from aruco_utils import get_aruco_dict, get_board
    config = load_configuration(configs.CONFIG_PATH)

    # Load aruco board for cv2 inference
    dictionary = get_aruco_dict(config.board_name)
    board = get_board(config)
    parameters = cv2.aruco.DetectorParameters_create()

    # Load models
    deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
    refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids=config.n_ids, device=device)

    if "DISPLAY" in os.environ:
        w = MagicGrid(1200, 1200, waitKey=0)

    # Inference test on custom image
    SAMPLE_IMAGES = './reference/samples_test/IMG_7412.png'
    import glob
    for p in glob.glob(SAMPLE_IMAGES):
        img = cv2.imread(p)

        # Run inference
        keypoints, out_img_dc = inf(img, config.n_ids, deepc, refinenet, device)
        print(keypoints)

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters)

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if "DISPLAY" in os.environ:
            if w.update([out_img_dc, out_img_cv]) == ord('q'):
                break


def inf(img, n_ids, deepc, refinenet, device):
    import time
    
    n = 100
    t = time.time()
    for i in range(n):
        keypoints, out_img_dc = infer_image(img, n_ids, deepc,
                                            refinenet,
                                            draw_pred=True,
                                            device=device)
    print(f"\033[95m--->INFERENCE TIME: {(time.time() - t)/n:0.4f} \033[0m")
    return keypoints, out_img_dc


if __name__ == '__main__':
    device='mps'
    inf_no_prof(device)