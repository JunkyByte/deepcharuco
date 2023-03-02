import random
import json
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from models.model_utils import pre_bgr_image
from gridwindow import MagicGrid
from data import inbound
from dataclasses import replace


def create_sample(image: np.ndarray, keypoints: np.ndarray):
    # Construct corner label
    # 1 pick a random point in the 64x64 window around the kpt TODO: check in bounds + padding (on correct side)
    # Create 192x192 patch around that arbitrary point
    # Rescale everything to 24x24 but keep label in 64x64

    off_x = random.randint(0, 63)  # random.randint includes BOTH endpoints
    off_y = random.randint(0, 63)  # TODO check me
    center_x = int(keypoints[0]) + off_x
    center_y = int(keypoints[1]) + off_y
    w_half = 192 // 2

    patch = image[center_y - w_half:center_y + w_half, center_x - w_half:center_x + w_half]

    print(patch.shape)
    # Pad each side
    lpad = max(0, w_half - center_y)
    rpad = max(0, w_half - center_y)
    print(lpad)
    # TODO

    corner = off_x + 64 * off_y  # corner offset wrt the center is the label
    print(patch.shape)
    return patch, corner


class RefineDataset(Dataset):
    def __init__(self, configs, labels, images_folder, validation=False, visualize=False):
        super().__init__()
        # Refinenet works on double resolution
        configs = replace(configs, input_size=(320 * 12, 240 * 12))  # TODO hardcoded?
        self._images_folder = images_folder
        self._visualize = visualize
        with open(labels, 'r') as f:
            self.labels = json.load(f)
        self.labels = self.labels['images']

        seed = 42 if validation else None
        if seed is not None:
            random.seed(seed)
        self.transform = Transformation(configs, negative_p=0, refinenet=True, seed=seed)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self._images_folder, label['file_name']), cv2.IMREAD_COLOR)

        # Apply pipeline of transformations
        image, keypoints, _ = self.transform(image).values()

        patch_resized = []
        corners = []
        for keypoint in keypoints:
            patch, corner = create_sample(image, keypoint)
            corners.append(corner)

            if self._visualize:
                corner_x = patch.shape[0] // 2 - corner % 64
                corner_y = patch.shape[0] // 2 - int(corner / 64)
                print(keypoint, corner)

                patch_copy = patch.copy()
                cv2.circle(patch_copy, (corner_x, corner_y), radius=3, color=(0, 255, 0),
                           thickness=2)

                patch_resized.append(cv2.resize(patch, (24, 24),
                                                interpolation=cv2.INTER_AREA))

        if self._visualize:
            w = MagicGrid(640, 640, waitKey=0)
            from aruco_utils import draw_inner_corners
            image_copy = draw_inner_corners(image, [keypoint], draw_ids=True, radius=3, color=(0, 0, 255))
            image_resized = cv2.resize(image_copy, (320, 240),
                                       interpolation=cv2.INTER_LANCZOS4)
            if w.update([image_resized, patch_copy, *patch_resized]) == ord('q'):
                import sys
                sys.exit()

        image = pre_bgr_image(image)
        sample = {
            'image': image,
            'label': corner,
        }

        return sample

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import configs
    from configs import load_configuration
    config = load_configuration(configs.CONFIG_PATH)
    dataset = RefineDataset(config,
                            config.train_labels,
                            config.train_images,
                            visualize=True,
                            validation=False)

    dataset_val = RefineDataset(config,
                                config.val_labels,
                                config.val_images,
                                visualize=True,
                                validation=True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    for i, b in enumerate(train_loader):
        print(i)

    for i, b in enumerate(val_loader):
        print(i)
