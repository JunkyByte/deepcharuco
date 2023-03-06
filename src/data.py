import random
import json
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from models.model_utils import pre_bgr_image
from gridwindow import MagicGrid


def create_label(image: np.ndarray, keypoints: np.ndarray, kpts_ids:
                 np.ndarray, isnegative: bool, dust_bin_ids: int):
    # Construct labels of the sample
    # loc head label
    dust_bin_loc = 8 * 8  # 8x8 spatial region (+ 1 for dust bin but we count from 0)
    loc = np.full((image.shape[0] // 8, image.shape[1] // 8), dtype=int,
                  fill_value=dust_bin_loc)

    # Ids head label
    bound_y_ds = image.shape[0] // 8
    bound_x_ds = image.shape[1] // 8
    ids = np.full((bound_y_ds, bound_x_ds), dtype=int, fill_value=dust_bin_ids)

    if isnegative:
        return loc, ids

    for keypoint, idx in zip(keypoints, kpts_ids):
        assert inbound(keypoint[0], keypoint[1], image.shape[1], image.shape[0]), keypoint

        # As we did downscaling these are float values of pixels, round them in bounds
        kx = keypoint[0]
        ky = keypoint[1]

        x = np.clip(int(kx / 8), 0, bound_x_ds - 1)
        y = np.clip(int(ky / 8), 0, bound_y_ds - 1)

        offset_x = int(kx) % 8
        offset_y = int(ky) % 8

        # If two occurences on same location / ids
        if ids[y, x] != dust_bin_ids:
            # print(f'Location of keypoint {idx} is already bounded to corner {ids[y, x]}')
            if random.random() > 0.5:  # At most 2 occ. => 50% swap
                continue

        loc[y, x] = offset_x + 8 * offset_y  # encode position of the pixels
        ids[y, x] = idx
    return loc, ids


class CharucoDataset(Dataset):
    def __init__(self, configs, labels, images_folder, validation=False, visualize=False):
        super().__init__()
        self.configs = configs
        self._images_folder = images_folder
        self._visualize = visualize
        with open(labels, 'r') as f:
            self.labels = json.load(f)
        self.labels = self.labels['images']

        seed = 42 if validation else None
        self.transform = Transformation(configs, negative_p=0.05, refinenet=False, seed=seed)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self._images_folder, label['file_name']), cv2.IMREAD_COLOR)

        # Apply pipeline of transformations
        image, keypoints, kpts_ids, isnegative = self.transform(image).values()

        dust_bin_ids = self.configs.n_ids
        loc, ids = create_label(image, keypoints, kpts_ids, isnegative, dust_bin_ids)

        if self._visualize:
            w = MagicGrid(640, 640, waitKey=0)
            from aruco_utils import draw_inner_corners, draw_circle_pred
            img = image.copy()
            img = draw_inner_corners(img, keypoints, kpts_ids, draw_ids=True, radius=3)
            img = draw_circle_pred(img, loc, ids, dust_bin_ids, draw_ids=True)

            if w.update([img]) == ord('q'):
                import sys
                sys.exit()

        image = pre_bgr_image(image)
        sample = {
            'image': image,
            'label': (loc, ids),
        }

        return sample

    def __len__(self):
        return len(self.labels)


def inbound(x, y, width, height):
    return x >= 0 and y >= 0 and x < width and y < height


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import configs
    from configs import load_configuration
    config = load_configuration(configs.CONFIG_PATH)
    dataset = CharucoDataset(config,
                             config.train_labels,
                             config.train_images,
                             visualize=True,
                             validation=False)

    dataset_val = CharucoDataset(config,
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
