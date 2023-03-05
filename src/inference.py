import cv2
from gridwindow import MagicGrid
import numpy as np

from aruco_utils import draw_circle_pred, draw_inner_corners, get_aruco_dict, get_board
from typing import Optional
import configs
from configs import load_configuration
from data import CharucoDataset
from models.model_utils import pred_to_keypoints, pred_sub_pix, extract_patches, pre_bgr_image
from models.net import lModel, dcModel
from models.refinenet import RefineNet, lRefineNet


def cv2_aruco_detect(image, dictionary, board, parameters):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(image, board, corners, ids, np.array([]))


    # If markers are detected, draw them and the board inner corners
    if len(corners) > 0:
        # Get board corners
        board_corners = [corners[i][0] for i in range(len(corners))]
        board_corners = np.array(board_corners, dtype=np.float32)

        # Draw board inner corners
        image = draw_inner_corners(image, board_corners.reshape((-1, 2)), ids=np.arange(board_corners.shape[0]))

    return image, corners, ids


def infer_image(img: np.ndarray, dust_bin_ids: int, deepc: lModel,
                refinenet: Optional[lRefineNet] = None,
                cv2_subpix: bool = False, draw_raw_pred: bool = False,
                draw_pred: bool = True):
    """
    Do full inference on a BGR image
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    loc_hat, ids_hat = deepc.infer_image(img_gray)
    kps_hat, ids_found = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)

    # Draw predictions in RED
    if draw_raw_pred or refinenet is None:
        img = draw_inner_corners(img, kps_hat, ids_found, radius=3,
                                 draw_ids=True, color=(0, 0, 255))

    if ids_found.shape[0] == 0:
        return np.array([]), img

    if refinenet is not None:
        patches = extract_patches(img_gray, kps_hat)

        # Extract 8x refined corners (in original resolution)
        # TODO we already computed preprocessing in deepc.infer_image
        patches = np.array([pre_bgr_image(p, is_gray=True) for p in patches])
        refined_kpts, _ = refinenet.infer_patches(patches, kps_hat)

        # Draw refinenet refined corners in yellow
        if draw_pred:
            img = draw_inner_corners(img, refined_kpts, ids_found,
                                     draw_ids=False, radius=1, color=(0, 255, 255))

    if cv2_subpix:
        ref_kps_cv2 = pred_sub_pix(infer_image, kps_hat, ids_found, region=(6, 6))
        # Draw cv2 refined corners in green
        img = draw_inner_corners(img, ref_kps_cv2, ids_found, draw_ids=False,
                                 radius=1, color=(0, 255, 0))

    # TODO: Make me beautiful
    keypoints = refined_kpts if refinenet else kpts_hat
    keypoints = np.array([[k[0], k[1], idx] for k, idx in sorted(zip(keypoints,
                                                                     ids_found),
                                                                 key=lambda x:
                                                                 x[1])])
    return keypoints, img


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)

    # Load aruco board for cv2 inference
    dictionary = get_aruco_dict(config.board_name)
    board = get_board(config)
    parameters = cv2.aruco.DetectorParameters_create()

    # Load models
    deepc = lModel.load_from_checkpoint("./reference/second-epoch-52-step=98k.ckpt",
                                        dcModel=dcModel(config.n_ids))
    deepc.eval()

    use_refinenet = True
    if use_refinenet:
        refinenet = lRefineNet.load_from_checkpoint("./reference/first-refinenet-epoch-59-step=282k.ckpt",
                                                    refinenet=RefineNet())
        refinenet.eval()

    # Inference test on validation data
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    w = MagicGrid(1000, 1000, waitKey=0)
    for ith, sample in enumerate(dataset_val):
        image, label = sample.values()
        loc, ids = label

        # Images returned from dataset are normalized.
        img = ((image * 255) + 128).astype(np.uint8)
        img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

        # Run inference
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet, cv2_subpix=False,
                                            draw_raw_pred=True)
        print(np.array(keypoints))
        # Draw labels in BLUE
        # img = draw_circle_pred(img, loc, ids, config.n_ids, radius=1, draw_ids=False)

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters) 

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if w.update([out_img_dc, out_img_cv]) == ord('q'):
            break

    # Inference test on custom image
    SAMPLE_IMAGES = './reference/samples_test/'
    import glob
    import os
    for p in glob.glob(SAMPLE_IMAGES + '*.png'):
        img = cv2.imread(p)

        # Run inference
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet, cv2_subpix=False,
                                            draw_raw_pred=True)
        print(np.array(keypoints))

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters) 

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if w.update([out_img_dc, out_img_cv]) == ord('q'):
            break

    # Video test inference
    import cv2

    cap = cv2.VideoCapture(os.path.join(SAMPLE_IMAGES, 'video_test.mp4'))
    target_size = (320, 240)
    while True:
        for i in range(3):
            ret, frame = cap.read()
            if not ret:
                break
            ratio = frame.shape[1] / frame.shape[0]
            image = cv2.resize(frame, (320, int(240 / ratio)))
        if not ret:
            break

        # Add padding to the resized frame if necessary
        if image.shape[0] < target_size[1]:
            padding = ((target_size[1] - image.shape[0]) // 2,
                       (target_size[1] - image.shape[0] + 1) // 2,
                       (target_size[0] - image.shape[1]) // 2,
                       (target_size[0] - image.shape[1] + 1) // 2)
            img = cv2.copyMakeBorder(image, *padding, borderType=cv2.BORDER_CONSTANT, value=0)

        # Run inference
        keypoints, out_img_dc = infer_image(img, config.n_ids, deepc,
                                            refinenet, cv2_subpix=False,
                                            draw_raw_pred=True)
        print(np.array(keypoints))

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters) 

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if w.update([out_img_dc, out_img_cv]) == ord('q'):
            break

    cap.release()
