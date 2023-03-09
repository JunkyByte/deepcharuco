import glob
import cv2
import configs
import numpy as np
from configs import load_configuration
from inference import infer_image, load_models
from aruco_utils import draw_inner_corners, board_image, get_board
from gridwindow import MagicGrid


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)
    board = get_board(config)
    img, corners = board_image(board, (480, 480),
                               config.row_count, config.col_count)
    img = draw_inner_corners(img, corners, np.arange(config.n_ids), draw_ids=True)

    num_corners_x = config.col_count
    num_corners_y = config.row_count

    calib_data = np.load('../data_demo/calib_frames/camera_params.npz')
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['distortion_coeffs']

    inn_rc = np.arange(1, num_corners_y)
    inn_cc = np.arange(1, num_corners_x)
    object_points = np.zeros(((num_corners_x - 1) * (num_corners_y - 1), 3), np.float32)
    object_points[:, :2] = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * config.square_len

    # Run inference
    deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
    refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids=config.n_ids)

    w = MagicGrid(640, 480, waitKey=1)
    for f in sorted(glob.glob('../data_demo/test_frames/*.png'))[::10]:
        img = cv2.imread(f)
        keypoints, img = infer_image(img, config.n_ids, deepc,
                                            refinenet, cv2_subpix=False,
                                            draw_raw_pred=True)
        print(keypoints)

        image_points = keypoints[:, :2].astype(np.float32)
        object_points_found = object_points[keypoints[:, 2].astype(int)]

        _, rvec, tvec = cv2.solvePnP(object_points_found, image_points, camera_matrix, dist_coeffs)

        # Draw axis
        axis_length = 0.02

        origin_point = np.zeros((3, 1))
        x_axis_point = np.array([[axis_length], [0], [0]])
        y_axis_point = np.array([[0], [axis_length], [0]])
        z_axis_point = np.array([[0], [0], [-axis_length]])
        axis_points = np.hstack([origin_point, x_axis_point, y_axis_point, z_axis_point])

        image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = np.int32(image_points.squeeze())

        cv2.line(img, tuple(image_points[0]), tuple(image_points[1]), (0, 0, 255), 2) # x-axis (red)
        cv2.line(img, tuple(image_points[0]), tuple(image_points[2]), (0, 255, 0), 2) # y-axis (green)
        cv2.line(img, tuple(image_points[0]), tuple(image_points[3]), (255, 0, 0), 2) # z-axis (blue)

        if w.update([img]) == ord('q'):
            import sys
            sys.exit()
