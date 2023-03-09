import cv2
from gridwindow import MagicGrid
from dataclasses import replace
import numpy as np

from aruco_utils import draw_inner_corners, get_aruco_dict, get_board, label_to_keypoints
from typing import Optional
import configs
from configs import load_configuration
from data import CharucoDataset
from models.model_utils import pred_to_keypoints, pred_sub_pix, extract_patches, pre_bgr_image
from models.net import lModel, dcModel
from models.refinenet import RefineNet, lRefineNet


def compute_l2_distance(keypoints, ids, target_keypoints, target_ids):
    # Initialize an empty array to store the distances
    distances = np.zeros((len(target_ids),))

    if distances.size == 0:
        return None

    # Loop over each unique id in target_ids
    for i, id in enumerate(np.unique(target_ids)):
        # Find the indices of keypoints and target_keypoints with the same id
        mask = np.nonzero(ids == id)[0]
        target_mask = np.nonzero(target_ids == id)[0]

        # If there are no matching keypoints, skip to the next id
        if mask.size == 0 or target_mask.size == 0:
            continue

        # Compute the L2 distance between the matching keypoints and target_keypoints
        dist = np.linalg.norm(keypoints[mask] - target_keypoints[target_mask], ord=2, axis=1)
        max_dist = np.max(dist)

        # Store the maximum distance for this id
        distances[i] = max_dist

    return distances


def pixel_error(kpts_raw, kpts_ref, kpts_target):
    if not set(kpts_raw[:, 2]).issubset(set(kpts_target[:, 2])):
        return None, None
    d = compute_l2_distance(kpts_raw[:, :2], kpts_raw[:, 2],
                            kpts_target[:, :2], kpts_target[:, 2])
    d_ref = compute_l2_distance(kpts_ref[:, :2], kpts_ref[:, 2],
                                kpts_target[:, :2], kpts_target[:, 2])
    d_raw_ref = compute_l2_distance(kpts_ref[:, :2], kpts_ref[:, 2],
                                    kpts_raw[:, :2], kpts_raw[:, 2])

    found = np.unique(kpts_raw[:, 2])
    print(f'Errors in pixels of the {len(found)}/{len(kpts_target[:, 2])} kpts found:')
    # for ith, id in enumerate(kpts_target[:, 2]):
    #     if id not in found:
    #         continue
    #     print(f'Marker {id:<2} Error Raw: {d[ith]:<5.3f} Error Refinet: {d_ref[ith]:<5.3f}')
    print(f'Mean error raw: {d.mean():<5.3f} Max error raw: {d.max():<5.3f}')
    print(f'Mean error ref: {d_ref.mean():<5.3f} Max error ref: {d_ref.max():<5.3f}')
    print(f'Mean dist raw/ref: {d_raw_ref.mean():<5.3f} Max dist raw/ref: {d_raw_ref.max():<5.3f}')
    return d.mean(), d_ref.mean()


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
    kpts_hat, ids_found = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)

    # Draw predictions in RED
    if draw_raw_pred:
        img = draw_inner_corners(img, kpts_hat, ids_found, radius=3,
                                 draw_ids=True, color=(0, 0, 255))

    if ids_found.shape[0] == 0:
        return np.array([]), img

    if refinenet is not None:
        patches = extract_patches(img_gray, kpts_hat)

        # Extract 8x refined corners (in original resolution)
        # TODO Optimization, we already computed preprocessing in deepc.infer_image
        patches = np.array([pre_bgr_image(p, is_gray=True) for p in patches])
        refined_kpts, _ = refinenet.infer_patches(patches, kpts_hat)

        # Draw refinenet refined corners in yellow
        if draw_pred:
            img = draw_inner_corners(img, refined_kpts, ids_found,
                                     draw_ids=False, radius=1, color=(0, 255, 255))

    if cv2_subpix:
        ref_kpts_cv2 = pred_sub_pix(infer_image, kpts_hat, ids_found, region=(6, 6))
        # Draw cv2 refined corners in green
        img = draw_inner_corners(img, ref_kpts_cv2, ids_found, draw_ids=False,
                                 radius=1, color=(0, 255, 0))

    keypoints = refined_kpts if refinenet else kpts_hat
    keypoints = np.array([[k[0], k[1], idx] for k, idx in sorted(zip(keypoints,
                                                                     ids_found),
                                                                 key=lambda x:
                                                                 x[1])])
    return keypoints, img


def load_models(deepc_ckpt: str, refinenet_ckpt: Optional[str] = None, n_ids: int = 16):
    deepc = lModel.load_from_checkpoint(deepc_ckpt, dcModel=dcModel(n_ids))
    deepc.eval()

    refinenet = None
    if refinenet_ckpt is not None:
        refinenet = lRefineNet.load_from_checkpoint(refinenet_ckpt, refinenet=RefineNet())
        refinenet.eval()

    return deepc, refinenet


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)

    # Load aruco board for cv2 inference
    dictionary = get_aruco_dict(config.board_name)
    board = get_board(config)
    parameters = cv2.aruco.DetectorParameters_create()

    # Load models
    deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
    refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
    deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids=config.n_ids)

    # Inference test on validation data
    up_scale = 1  # Set to 8 for with/without refinenet comparison
    if up_scale > 1:
        config = replace(config, input_size = (320 * up_scale, 240 * up_scale))

    # Load val dataset
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

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
                                            refinenet, cv2_subpix=False,
                                            draw_pred=True, draw_raw_pred=True)
        print('Keypoints\n', keypoints)

        if up_scale > 1:
            # Run inference again without refinenet
            keypoints_raw, _ = infer_image(img, config.n_ids, deepc, None,
                                           cv2_subpix=False,
                                           draw_raw_pred=False,
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
        print(keypoints)

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
        print(keypoints)

        # cv2 inference
        out_img_cv, corners, _ = cv2_aruco_detect(img.copy(), dictionary, board, parameters)

        # show result
        out_img_dc = cv2.resize(out_img_dc, (out_img_dc.shape[1] * 3, out_img_dc.shape[0] * 3), cv2.INTER_LANCZOS4)
        out_img_cv = cv2.resize(out_img_cv, (out_img_cv.shape[1] * 3, out_img_cv.shape[0] * 3), cv2.INTER_LANCZOS4)
        if w.update([out_img_dc, out_img_cv]) == ord('q'):
            break

    cap.release()
