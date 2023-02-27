import copy
import json
import math
import os

import cv2
import numpy as np
import pycocotools
import pycocotools.mask

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from gridwindow import MagicGrid


class CharucoDataset(Dataset):
    def __init__(self, configs, labels, images_folder, visualize=False):
        super().__init__()
        self.configs = configs
        self._images_folder = images_folder
        self._visualize = visualize
        with open(labels, 'r') as f:
            self.labels = json.load(f)
        self.labels = self.labels['images']
        self.transform = Transformation(configs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self._images_folder, label['file_name']), cv2.IMREAD_COLOR)

        # Apply pipeline of transformations
        image, keypoints = self.transform(image).values()  # TODO

        sample = {  # TODO
            'label': label,  # This is not actual label
            'image': image,
        }

        w = MagicGrid(640, 640, waitKey=0)
        if self._visualize:
            from aruco_utils import draw_inner_corners
            img = sample['image'].copy()
            img = draw_inner_corners(img, keypoints, draw_ids=True)

            if w.update([img]) == ord('q'):
                import sys
                sys.exit()

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256  # TODO
        image = image.transpose((2, 0, 1))
        sample['image'] = image

        del sample['label']
        return sample

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import configs
    from configs import load_configuration
    configs = load_configuration(configs.CONFIG_PATH)
    dataset = CharucoDataset(configs,
                             '../data_demo/annotations/captions_val2017.json',
                             '../data_demo/val2017/',
                             visualize=True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, b in enumerate(train_loader):
        print(i)
