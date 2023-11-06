import cv2
from dataclasses import replace
import numpy as np
import torch

from aruco_utils import draw_inner_corners, label_to_keypoints, cv2_aruco_detect
from typing import Optional
import configs
from configs import load_configuration
from inference import load_models, infer_image


if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

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

    # Inference test on custom image
    SAMPLE_IMAGE = './reference/samples_test/IMG_7412.png'
    img = cv2.imread(SAMPLE_IMAGE)

    # Warmup
    for i in range(5):
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet,
                                            draw_pred=False,
                                            device=device)

    # Run inference
    import time
    n = 500
    t = time.time()
    for i in range(n):
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet,
                                            draw_pred=False,
                                            device=device)
    print(f"\033[95m--->FPS: {n/(time.time() - t):0.1f} \033[0m")