import random
import json
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from gridwindow import MagicGrid


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
        self.transform = Transformation(configs, seed=seed)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self._images_folder, label['file_name']), cv2.IMREAD_COLOR)

        # Apply pipeline of transformations
        image, keypoints = self.transform(image).values()  # TODO

        # Construct labels of the sample
        # loc head label
        dust_bin_loc = 8 * 8  # 8x8 spatial region (+ 1 for dust bin but we count from 0)
        loc = np.full((image.shape[0] // 8, image.shape[1] // 8), dtype=int,
                      fill_value=dust_bin_loc)

        # Ids head label
        bound_y_ds = image.shape[0] // 8
        bound_x_ds = image.shape[1] // 8
        dust_bin_ids = self.configs.n_ids
        ident = np.full((bound_y_ds, bound_x_ds), dtype=int, fill_value=dust_bin_ids)

        for ith, keypoint in enumerate(keypoints):
            if not inbound(keypoint[0], keypoint[1], image.shape[1], image.shape[0]):
                # print(f'Keypoint {ith} is out of bound {keypoint}')
                continue

            # As we did downscaling these are float values of pixels, round them in bounds
            kx = keypoint[0]
            ky = keypoint[1]

            x = np.clip(int(kx / 8), 0, bound_x_ds - 1)
            y = np.clip(int(ky / 8), 0, bound_y_ds - 1)

            offset_x = int(kx) % 8
            offset_y = int(ky) % 8

            # If two occurences on same location / identity
            if ident[y, x] != dust_bin_ids:
                # print(f'Location of keypoint {ith} is already bounded to corner {ident[y, x]}')
                if random.random() > 0.5:  # At most 2 occ. => 50% swap TODO
                    continue

            loc[y, x] = offset_x + 8 * offset_y  # encode position of the pixels
            ident[y, x] = ith

        if self._visualize:
            w = MagicGrid(640, 640, waitKey=0)
            from aruco_utils import draw_inner_corners, draw_circle_pred
            img = image.copy()
            img = draw_inner_corners(img, keypoints, draw_ids=True, radius=3)
            img = draw_circle_pred(img, loc, ident, dust_bin_ids, draw_ids=True)

            if w.update([img]) == ord('q'):
                import sys
                sys.exit()

        image = image.astype(np.float32)
        image = (image - 128) / 256  # TODO
        image = image.transpose((2, 0, 1))

        sample = {  # TODO
            'image': image,
            'label': (loc, ident),
        }

        return sample

    def __len__(self):
        return len(self.labels)


def inbound(x, y, width, height):
    return x > 0 and y > 0 and x < width and y < height


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import configs
    from configs import load_configuration
    configs = load_configuration(configs.CONFIG_PATH)
    dataset = CharucoDataset(configs,
                             '../data_demo/annotations/captions_val2017.json',
                             '../data_demo/val2017/',
                             visualize=True,
                             validation=False)

    dataset_val = CharucoDataset(configs,
                                 '../data_demo/annotations/captions_val2017.json',
                                 '../data_demo/val2017/', visualize=True,
                                 validation=True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    for i, b in enumerate(train_loader):
        print(i)

    for i, b in enumerate(val_loader):
        print(i)
