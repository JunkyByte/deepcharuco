import random
import json
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from model_utils import pre_bgr_image
from dataclasses import replace


def create_sample(image: np.ndarray, up_factor, keypoints: np.ndarray):
    # Construct corner label
    w_half = (192 + 64) // (2 * up_factor)  # TODO

    center_x = int(keypoints[0])
    center_y = int(keypoints[1])

    # Take a patch
    # if up_factor > 1 we need to take full region, resize it by up_factor and then continue
    # Take centered patch -> upscale
    patch_og_res = image[center_y - w_half:center_y + w_half,
                         center_x - w_half:center_x + w_half]
    if not patch_og_res.shape == (256 // up_factor, 256 // up_factor, 3):
        return None, None

    # Upscale this patch
    patch_up = cv2.resize(patch_og_res, (192 + 64, 192 + 64), cv2.INTER_CUBIC)

    # Now apply translation
    tl = 32
    off_x = random.randint(-tl, tl - 1)  # random.randint includes BOTH endpoints
    off_y = random.randint(-tl, tl - 1)  # TODO check me, also use gaussian? cornerSubPix?

    new_center_x = patch_up.shape[1] // 2 + off_x
    new_center_y = patch_up.shape[0] // 2 + off_y

    patch_new = patch_up[new_center_y - 96:new_center_y + 96,
                         new_center_x - 96:new_center_x + 96]
    patch = cv2.resize(patch_new, (24, 24), cv2.INTER_AREA)

    # TODO: Notice the -1 check!
    corner_x = -off_x + tl - 1  # Calculate the pixel 'number' (starting from 0 on top left of the 64x64 region)
    corner_y = -off_y + tl - 1  # Notice the '-' in front is because we have to invert the translation we just did

    corner = corner_x + 64 * corner_y  # corner offset wrt the center is the label
    return patch, corner


class RefineDataset(Dataset):
    def __init__(self, configs, labels, images_folder, validation=False, visualize=False, total=1):
        super().__init__()
        self.s_factor = 2
        self.zoom = 1
        self.total = total
        configs = replace(configs, input_size=(320 * self.s_factor * self.zoom,
                                               240 * self.s_factor * self.zoom))
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
        image, keypoints, *_ = self.transform(image).values()

        patch_resized = []  # Just for visualization

        corners = []
        patches = []
        up_factor = 8 // self.s_factor
        random.shuffle(keypoints)
        for keypoint in keypoints:
            patch, corner = create_sample(image, up_factor, keypoint)
            if patch is None:  # Pad not implemented, sometimes there are regions not big enough
                continue

            patches.append(patch)
            corners.append(corner)
            assert 0 <= corner < 4096, corner

            # Just visualization
            if self._visualize:
                corner_x = corner % 64  # These are in 8x scale
                corner_y = int(corner / 64)

                # We need to move to central 64x64 region position (is offset wrt that)
                corner_x = corner_x + 64
                corner_y = corner_y + 64

                # Visualization in original resolution is poor, visualize in 8x patch!
                patch_vis = cv2.resize(patch, (192, 192), cv2.INTER_CUBIC)
                patch_vis = cv2.circle(patch_vis, (corner_x, corner_y),
                                       radius=5, color=(0, 255, 0),
                                       thickness=2)

                patch_vis = cv2.circle(patch_vis, (patch_vis.shape[1] // 2, patch_vis.shape[0]// 2),
                                       radius=5, color=(0, 0, 255),
                                       thickness=2)
                patch_resized.append(patch_vis)

            if len(patches) == self.total:
                break  # Done!

        if self._visualize:
            from gridwindow import MagicGrid
            w = MagicGrid(640, 640, waitKey=0)
            if w.update([*patch_resized]) == ord('q'):
                import sys
                sys.exit()

        patches = [pre_bgr_image(p) for p in patches]

        # Must have same length for each batch.
        # We duplicate samples to reach correct numbering
        missing = self.total - len(corners)
        for _ in range(missing):
            idx = random.randint(0, len(patches) - 1)
            patches.append(patches[idx])
            corners.append(corners[idx])

        return patches, corners

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
