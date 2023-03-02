import sys
import cv2
from gridwindow import MagicGrid
import numpy as np
import torch

from aruco_utils import draw_circle_pred, draw_inner_corners
import configs
from configs import load_configuration
from data import CharucoDataset
from models.model_utils import pre_bgr_image, label_to_keypoints, pred_argmax, pred_sub_pix
from models.net import lModel, dcModel


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)
    model = lModel.load_from_checkpoint("./reference/epoch=44-step=83205.ckpt",
                                        dcModel=dcModel(config.n_ids))
    model.eval()

    
    # Inference test on validation data
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    w = MagicGrid(1200, 1200, waitKey=0)
    for ith, sample in enumerate(dataset_val):
        image, label = sample.values()
        loc, ids = label

        # Images returned from dataset are normalized.
        img = ((image.copy() * 255) + 128).astype(np.uint8)
        img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)
        infer_image = img.copy()

        # Draw labels in BLUE
        # img = draw_circle_pred(img, loc, ids, config.n_ids, radius=3, draw_ids=False)

        # Do prediction
        loc_hat, ids_hat = model.infer_image(infer_image)

        # Draw predictions in RED  # TODO fix this
        loc_hat = loc_hat[0]
        ids_hat = ids_hat[0]
        img = draw_circle_pred(img, loc_hat, ids_hat, config.n_ids, radius=1,
                               draw_ids=True, color=(0, 0, 255))

        sub_pix_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
        ref_corners, ids = pred_sub_pix(sub_pix_image, loc_hat, ids_hat, config.n_ids, region=(5, 5))
        img = draw_inner_corners(img, ref_corners, draw_ids=False, radius=1, color=(0, 255, 0))

        # Show result
        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), cv2.INTER_LANCZOS4)
        if w.update([img]) == ord('q'):
            break

    
    # Inference test on custom image
    SAMPLE_IMAGES = './reference/samples_test/'
    import glob, os
    for p in glob.glob(SAMPLE_IMAGES + '*.png'):
        image = cv2.imread(p)
        loc_hat, ids_hat = model.infer_image(image)

        # Draw predictions in RED
        loc_hat = loc_hat[0]
        ids_hat = ids_hat[0]
        image = draw_circle_pred(image, loc_hat, ids_hat, config.n_ids, radius=1,
                               draw_ids=True, color=(0, 0, 255))

        if w.update([image]) == ord('q'):
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

        loc_hat, ids_hat = model.infer_image(image)

        # Draw predictions in RED
        loc_hat = loc_hat[0]
        ids_hat = ids_hat[0]
        image = draw_circle_pred(image, loc_hat, ids_hat, config.n_ids, radius=1,
                               draw_ids=True, color=(0, 0, 255))

        if w.update([image]) == ord('q'):
            break

    cap.release()
