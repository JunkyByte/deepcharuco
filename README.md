# deepcharuco

This repository is an unofficial implementation of the model proposed by Hu et al. in their paper [Deep ChArUco: Dark ChArUco Marker Pose Estimation](https://arxiv.org/abs/1812.03247), CVPR-2019 for ChArUco board localization.
This is a personal project and I have no affiliation with the authors, the results obtained might differ and should not be considered as a reference for the paper results. All the data used by the paper is not public and therefore a fair comparison is impossible.

Some implementation details were not thoroughly discussed in the paper and I did my best to obtain comparable results usinig a similar model architecture and training procedure. I trained both the deep charuco and refinenet models on synthetic data generated on the fly. To train COCO images are required, further details in training section.

## Overview and results
![architecture](https://i.imgur.com/W8TnGgm.png)

## Setup (training / inference)
To train COCO images and annotations are needed (it just needs to contain images informations, I use `captions_*.json`).
To download coco images and labels refer to (https://cocodataset.org/#download) or use the following
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```
If you just want to run inference on COCO validation you can download only the `val2017` images and annotations.  
To specify the paths used and other configurations we use a `yaml` config file.
By default the scripts will try to load `src/config.yaml`. There's a `demo_config.yaml` that you can copy, the parameters are the following:
```yaml
input_size: [320, 240]  # Input images size used for training (width, height)
bs_train: 32  # batch size for deep charuco
bs_train_rn: 64  # batch size for refinenet
bs_val: 64  # batch size for deep charuco in validation
bs_val_rn: 128  # batch size for refinenet in validation
num_workers: 6  # num of workers used by dataloader in training
train_labels: '/home/adryw/dataset/coco25/annotations/captions_train2017.json'  # training labels
val_labels: '/home/adryw/dataset/coco25/annotations/captions_val2017.json'  # validation labels
val_images: '/home/adryw/dataset/coco25/val2017/'  # training images
train_images: '/home/adryw/dataset/coco25/train2017/'  # validation images
```

## Inference
If you setup the val data you can run `inference.py` for a preview of results on validation data used in training. To do inference on your images check `inference.py` code and adapt it to your needs, the idea is the following:
```python
import cv2
from inference import infer_image, load_models

# Load models
deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"
n_ids = 16  # The number of corners (models pretrained use 16 for default board)
deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids)

# Run inference on BGR image
img = cv2.imread(p)

# The out_img will have corners plotted on it if draw_pred is True
# The keypoints format is (x, y, id_corner)
keypoints, out_img = infer_image(img, n_ids, deepc, refinenet, draw_pred=True)
```

## Training
The trained model provided for DeepCharuco network is associated to the following tensorboard plots during training
<details>
  <summary>Plots</summary>
  
  ![train_res](https://i.imgur.com/PFTL10P.png)
  ![train_res2](https://i.imgur.com/pdrC5C4.png)
  Where `l2_pixels` is the euclidean distance in pixels of the corners the model found during validation and `match_ratio` is the percentage of corners found over the total in each image. Please look at `models/metrics.py`, they are not perfect metrics but provide useful insights of the model training.
</details>


## Common issues
