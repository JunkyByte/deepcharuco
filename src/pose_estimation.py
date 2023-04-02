import glob
import argparse
import torch
import os
import cv2
import configs
import numpy as np
import tqdm
from configs import load_configuration
from utils import save_video
from inference import infer_image, load_models, solve_pnp
from aruco_utils import draw_inner_corners, board_image, get_board, cv2_aruco_detect, get_aruco_dict
from gridwindow import MagicGrid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="path to input image directory")
    args = parser.parse_args()

    config = load_configuration(configs.CONFIG_PATH)
    board = get_board(config)
    img, corners = board_image(board, (480, 480),
                               config.row_count, config.col_count)
    img = draw_inner_corners(img, corners, np.arange(config.n_ids), draw_ids=True)

    calib_data = np.load('../data_demo/calib_frames/camera_params.npz')
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['distortion_coeffs']

    # Run inference
    deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
    refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids=config.n_ids, device=device)

    # These are needed just for comparison with cv2
    dictionary = get_aruco_dict(config.board_name)
    parameters = cv2.aruco.DetectorParameters_create()

    frames = []
    if "DISPLAY" in os.environ:
        w = MagicGrid(640, 480, waitKey=1)
    for f in tqdm.tqdm(sorted(glob.glob(os.path.join(args.input_dir, '*.png')))):
        img = cv2.imread(f)
        img_og = img.copy()

        if img is None:
            raise ValueError("Could not read the images")
        keypoints, img = infer_image(img, config.n_ids, deepc, refinenet,
                                     draw_pred=True)

        ret, rvec, tvec = solve_pnp(keypoints, config.col_count,
                                    config.row_count, config.square_len,
                                    camera_matrix, dist_coeffs)

        if ret:  # Draw axis
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01, 2)

        # Do the same using aruco cv2 detect markers (this is for video comparison)
        out_img_cv, corners, ids = cv2_aruco_detect(img_og, dictionary, board, parameters)
        ret_cv2, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board,
                                                          camera_matrix,
                                                          dist_coeffs, None,
                                                          None)

        object_points_found = np.array(board.objPoints)[ids].reshape((-1, 3))
        image_points = np.array(corners).reshape((-1, 2))

        ret_cv2 = False
        if image_points.shape[0] >= 4:
            ret_cv2, rvec, tvec = cv2.solvePnP(object_points_found, image_points, camera_matrix, dist_coeffs)

        if ret_cv2:
            cv2.drawFrameAxes(out_img_cv, camera_matrix, dist_coeffs, rvec, tvec, 0.01, 2)

        frames.append(np.hstack([img, out_img_cv]))
        if "DISPLAY" in os.environ:
            if w.update([frames[-1]]) == ord('q'):
                import sys
                sys.exit()

    save_video(frames, os.path.join(args.input_dir, 'res.mp4'), fps=30)
