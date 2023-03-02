import random
import json
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from models.model_utils import pre_bgr_image
from dataclasses import replace


def create_sample(image: np.ndarray, up_factor, keypoints: np.ndarray):
    # Construct corner label
    w_half = (192 // up_factor + 64 // up_factor) // 2  # TODO

    center_x = int(keypoints[0])
    center_y = int(keypoints[1])

    # Take a patch
    # if up_factor > 1 we need to take full region, resize it by up_factor and then continue
    # Take centered patch -> upscale
    patch_og_res = image[center_y - w_half:center_y + w_half,
                         center_x - w_half:center_x + w_half]
    print('patch original shape', patch_og_res.shape)
    # TODO: padding here
    # Pad each side
    # lpad = max(0, w_half - center_y)
    # rpad = max(0, w_half - center_y)
    # print(lpad)
    # TODO

    # Upscale this patch
    patch_up = cv2.resize(patch_og_res, (192 + 64, 192 + 64), cv2.INTER_CUBIC)

    print('patch_up new shape', patch_up.shape)

    # Now apply translation
    off_x = random.randint(-33, 31)  # random.randint includes BOTH endpoints
    off_y = random.randint(-33, 31)  # TODO check me
    new_center_x = patch_up.shape[1] // 2 + off_x
    new_center_y = patch_up.shape[0] // 2 + off_y

    patch_up = cv2.circle(patch_up, (patch_up.shape[1] //2,
                                     patch_up.shape[0]//2), radius=3,
                          color=(255, 0, 0))
    print(new_center_x, new_center_y)
    patch_new = patch_up[new_center_y - 96:new_center_y + 96,
                         new_center_x - 96:new_center_x + 96]
    print('new patch_new after translation shape', patch_new.shape)
    
    patch = cv2.resize(patch_new, (24, 24), cv2.INTER_AREA)
    print('after resize', patch.shape)

    from gridwindow import MagicGrid
    w = MagicGrid(640, 640, waitKey=0)

    if w.update([patch_og_res, patch_up, patch_new, patch]) == ord('q'):
        import sys
        sys.exit()

    corner_x = off_x + 33
    corner_y = off_y + 33
    corner = corner_x + 64 * corner_y  # corner offset wrt the center is the label
    print(corner)
    return patch, corner


class RefineDataset(Dataset):
    def __init__(self, configs, labels, images_folder, validation=False, visualize=False):
        super().__init__()
        self.s_factor = 1
        configs = replace(configs, input_size=(320 * self.s_factor, 240 * self.s_factor))
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
        up_factor = 8 // self.s_factor
        for keypoint in keypoints:
            patch, corner = create_sample(image, up_factor, keypoint)
            corners.append(corner)

            if self._visualize:
                corner_x = patch.shape[1] // 2 - (corner % 64) // 2  # ? TODO
                corner_y = patch.shape[0] // 2 - int(corner / 64) // 2
                print(keypoint, corner, patch.shape)

                patch_copy = patch.copy()
                cv2.circle(patch_copy, (corner_x, corner_y), radius=3, color=(0, 255, 0),
                           thickness=2)

                patch_resized.append(cv2.resize(patch_copy, (24, 24),
                                                interpolation=cv2.INTER_AREA))

        if self._visualize:
            from gridwindow import MagicGrid
            w = MagicGrid(640, 640, waitKey=0)
            from aruco_utils import draw_inner_corners
            image_copy = draw_inner_corners(image, [keypoint], draw_ids=True, radius=3, color=(0, 0, 255))
            image_resized = cv2.resize(image_copy, (320, 240),
                                       interpolation=cv2.INTER_LANCZOS4)
            if w.update([image_resized, patch_copy, *patch_resized]) == ord('q'):
                import sys
                sys.exit()

        # TODO
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
