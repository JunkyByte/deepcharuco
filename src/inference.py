import cv2
from gridwindow import MagicGrid
import numpy as np

from aruco_utils import draw_circle_pred, draw_inner_corners
import configs
from configs import load_configuration
from data import CharucoDataset
from models.model_utils import label_to_keypoints, pred_sub_pix, extract_patches, pre_bgr_image
from models.net import lModel, dcModel
from models.refinenet import RefineNet, lRefineNet


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)
    deepc = lModel.load_from_checkpoint("./reference/second-epoch-52-step=98k.ckpt",
                                        dcModel=dcModel(config.n_ids))
    deepc.eval()

    use_refinenet = True
    if use_refinenet:
        refinenet = lRefineNet.load_from_checkpoint("./reference/epoch=3-step=17524.ckpt",
                                                    refinenet=RefineNet())
        refinenet.eval()

    # Inference test on validation data
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    w = MagicGrid(1200, 1200, waitKey=0)
    for ith, sample in enumerate(dataset_val):
        image, label, kpts_ids = sample.values()
        loc, ids = label

        # Images returned from dataset are normalized.
        img = ((image.copy() * 255) + 128).astype(np.uint8)
        img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

        # Do prediction  # TODO: Put together inference for deepc and refinenet
        infer_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        loc_hat, ids_hat = deepc.infer_image(infer_image)

        # Draw labels in BLUE
        # img = draw_circle_pred(img, loc, ids, config.n_ids, radius=3, draw_ids=False)

        # Draw predictions in RED  # TODO fix this
        img = draw_circle_pred(img, loc_hat, ids_hat, config.n_ids, radius=3,
                               draw_ids=True, color=(0, 0, 255))

        patches = []
        if use_refinenet:
            # TODO: This is computed twice with pred_sub_pix
            kps, pred_ids = label_to_keypoints(loc_hat, ids_hat, config.n_ids)

            if len(pred_ids):
                # This is also computed twice with deepc infer_image
                patches = extract_patches(infer_image, kps)
                patches_vis = patches.copy()

                patches = np.array([pre_bgr_image(p, is_gray=True) for p in patches])

                # Extract 8x refined corners (in original resolution)
                refined_kpts = refinenet.infer_patches(patches, kps)

                # Draw refinenet refined corners in blue
                img = draw_inner_corners(img, refined_kpts, pred_ids,
                                         draw_ids=False, radius=1, color=(255, 0, 0))

        ref_corners, pred_ids = pred_sub_pix(infer_image, loc_hat, ids_hat,
                                        config.n_ids, region=(5, 5))

        # Draw cv2 refined corners in green
        img = draw_inner_corners(img, ref_corners, pred_ids, draw_ids=False,
                                 radius=1, color=(0, 255, 0))

        # Show result
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), cv2.INTER_LANCZOS4)
        if w.update([img, *[p for p in patches_vis]]) == ord('q'):
            break

    # Inference test on custom image
    SAMPLE_IMAGES = './reference/samples_test/'
    import glob
    import os
    for p in glob.glob(SAMPLE_IMAGES + '*.png'):
        image = cv2.imread(p)
        loc_hat, ids_hat = deepc.infer_image(image)

        # Draw predictions in RED
        image = draw_circle_pred(image, loc_hat, ids_hat, config.n_ids,
                                 radius=1, draw_ids=True, color=(0, 0, 255))

        kps, _ = label_to_keypoints(loc_hat, ids_hat, config.n_ids)
        patches = [image[y - 8: y + 8, x - 8: x + 8] for (x, y) in kps]

        if w.update([image, *patches]) == ord('q'):
            break

    # Video test inference
    import cv2

    cap = cv2.VideoCapture(os.path.join(SAMPLE_IMAGES, 'video_test.mp4'))
    target_size = (320, 240)
    ret = True
    while True:
        for i in range(3):
            ret, frame = cap.read()
            image = cv2.resize(frame, target_size)
            if not ret:
                break

        # Add padding to the resized frame if necessary
        if image.shape[0] < target_size[1]:
            padding = ((target_size[1] - image.shape[0]) // 2, (target_size[1] - image.shape[0] + 1) // 2,
                       (target_size[0] - image.shape[1]) // 2, (target_size[0] - image.shape[1] + 1) // 2)
            image = cv2.copyMakeBorder(image, *padding, borderType=cv2.BORDER_CONSTANT, value=0)

        loc_hat, ids_hat = deepc.infer_image(image)

        # Draw predictions in RED
        image = draw_circle_pred(image, loc_hat, ids_hat, config.n_ids, radius=1,
                                 draw_ids=True, color=(0, 0, 255))

        if w.update([image]) == ord('q'):
            break

    cap.release()
