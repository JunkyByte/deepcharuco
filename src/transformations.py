import imgaug
import albumentations as A
import numpy as np
import cv2
import random

from aruco_utils import board_image, draw_inner_corners, get_board

from custom_aug.custom_aug import PasteBoard, HistogramMatching


class Transformation:
    """
    Class to apply augmentation on COCO dataset to train deepcharuco.
    Steps:
    0) Choose if is a negative sample, in that case just return an augmented coco
    1) Augment board image (+ mask + corners)
    2) Histogram matching of board image given coco image
    3) Paste image on coco image
    4) Augment coco + board image
    5) Profit!
    """
    def __init__(self, configs, negative_p=0.1, seed=None):
        self.seed = seed
        self.negative_p = negative_p
        if seed is not None:
            random.seed(seed)
            imgaug.random.seed(seed)

        min_r = min(configs.input_size)
        board = get_board(configs)
        board_img, corners = board_image(board, (min_r, min_r),
                                         configs.row_count, configs.col_count)

        self.board_img = board_img
        self.corners = corners
        self.ids = np.arange(self.corners.shape[0])
        self.board_mask = np.full((board_img.shape[0], board_img.shape[1]),
                                  dtype=np.uint8, fill_value=255)

        board_transf = [
            A.PadIfNeeded(min_height=configs.input_size[1],
                          min_width=configs.input_size[0],
                          always_apply=True, border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
            A.augmentations.geometric.Affine(
                scale=(0.2, 0.8),
                rotate=(-360, 360),
                shear=(-40, 40),
                translate_percent=(-0.45, 0.45),
                keep_ratio=True,
                fit_output=False,
                always_apply=True,
            ),
            A.Resize(height=configs.input_size[1], width=configs.input_size[0],
                     always_apply=True)
        ]

        # 1) Create transformation for board image
        self._transf_board = A.Compose(
            board_transf,
            keypoint_params=A.KeypointParams(format='xy',
                                             label_fields=['ids'],
                                             remove_invisible=False)
        )

        # 1bis) COCO transformation
        self._transf_coco = A.Compose([
            A.augmentations.geometric.Flip(p=0.5),
            A.augmentations.geometric.Rotate(limit=(-180, 180), crop_border=True, p=0.5),
            A.PadIfNeeded(min_height=configs.input_size[1],
                          min_width=configs.input_size[0], always_apply=True,
                          border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=0),
            A.augmentations.RandomCrop(height=configs.input_size[1],
                                       width=configs.input_size[0],
                                       always_apply=True),
        ])

        # 2 + 3) Apply histogram matching then Paste transformation
        self._transf_joint = A.Compose([
            HistogramMatching(blend_ratio=(0, 0.5), p=0.7),
            PasteBoard(always_apply=True),

            # Augmentations as from paper
            A.augmentations.GaussNoise(p=0.5),
            A.augmentations.MotionBlur(p=0.5),
            A.augmentations.GaussianBlur(p=0.25),
            A.augmentations.MultiplicativeNoise(p=0.5),
            A.augmentations.RandomBrightnessContrast(brightness_limit=(-0.8, 0.35),
                                                     contrast_limit=0, p=0.5),
            # A.augmentations.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_dimension=4, p=0.3),

        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['ids'],
                                            remove_invisible=False)
        )

    def _transform_board(self):
        t_res = self._transf_board(image=self.board_img,
                                   mask=self.board_mask,
                                   keypoints=self.corners,
                                   ids=self.ids)
        return t_res

    def __call__(self, coco_img):
        return self.transform(coco_img)

    def transform(self, coco_img):
        res = self._transform_board()  # Generate board image

        # Adapt coco image to input_size
        coco_img = self._transf_coco(image=coco_img)['image']

        # We also generate negative instances without board
        isnegative = random.random() < self.negative_p

        # Apply joint pipeline
        res = self._transf_joint(**res, target=coco_img, isnegative=isnegative)
        return {'image': res['image'], 'keypoints': res['keypoints'],
                'isnegative': isnegative}  # TODO return?


if __name__ == '__main__':
    from gridwindow import MagicGrid
    import configs
    from configs import load_configuration
    config = load_configuration(configs.CONFIG_PATH)
    t = Transformation(config)

    w = MagicGrid(640, 640, waitKey=0)
    # while True:
    #     t_res = t._transform_board()
    #     t_img, t_corners = t_res['image'], t_res['keypoints']
    #     t_ids, t_mask = t_res['ids'], t_res['mask']
    #     print(t_img.shape)

    #     t_img = draw_inner_corners(t_img, t_corners, draw_ids=True)
    #     if w.update([t_img, t_mask]) == ord('q'):
    #         break

    while True:
        t_res = t.transform(np.random.randn(*config.input_size[::-1], 3).astype(np.uint8))
        t_img, t_corners = t_res['image'], t_res['keypoints']

        t_img = draw_inner_corners(t_img, t_corners, draw_ids=True)
        if w.update([t_img]) == ord('q'):
            break
