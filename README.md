# deepcharuco

This repository is an unofficial implementation of the model proposed by Hu et al. in their paper [Deep ChArUco: Dark ChArUco Marker Pose Estimation CVPR2019](https://arxiv.org/abs/1812.03247) for ChArUco board localization.
This is a personal project and I have no affiliation with the authors, the results obtained might differ and should not be considered a reference of the paper results. All the data used by the paper is not public and therefore a fair comparison is impossible.

Some implementation details were not thoroughly discussed in the paper and I did my best to obtain comparable results using a similar model architecture and training procedure. I trained both the deep charuco and refinenet models on synthetic data generated on the fly. To train COCO images are required, further details in training section.

## Overview and results
![architecture](https://i.imgur.com/W8TnGgm.png)
The idea is to build a network which can localize charuco inner corners and recognize the ids of the corners found. The trained network is fit on a particular board configuration, in the case of the paper the board has `12` aruco markers for a total of 16 inner corners ([board image](src/reference/board_image_240x240.jpg)).
The network is divided into two parts, what we call `DeepCharuco` which is a fully convolutional network for localization and identification [net.py](src/models/net.py) and `RefineNet` another fully convolutional network for corner refinement [refinenet.py](src/models/refinenet.py). Refer to the paper for details.  

`DeepCharuco` takes a `(1, H, W)` grayscale image and outputs 2 tensors:
- loc: `(65, H/8, W/8)` representing the probability that a corner is in a particular pixel (`64` pixels for a `8x8` region + 1 dust bin channel)
- ids: `(17, H/8, W/8)` representing the probability that a certain `8x8` region contains a particular corner id + 1 dust bin channel.
`RefineNet` takes a `(1, 24, 24)` patch around a corner and outputs a `(1, 64, 64)` tensor representing the probability that the corner is in a certain (sub-)pixel in `8x` resolution of the central `8x8` region of the patch.

## Setup for training (and inference on val data)
`requirements.txt` should contain a valid list of requirements, notice `opencv-contrib` is `<4.7`.  
If you wan to run `inference.py` or the `data.py` and `data_refinenet.py` you will need to install [gridwindow](https://github.com/JunkyByte/python-gridwindow) which is used for visualization (or replace everything with normal `cv2` windows)
Synthetic data for `DeepCharuco` training
![data](https://i.imgur.com/KasncjL.png)  
Synthetic data for `RefineNet` training (red center of image, green target)
![data2](https://i.imgur.com/CveVxF0.png)  
COCO images and annotations are needed for training (the label used just needs to contain images informations, I use `captions_*.json`).
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
The inference uses pytorch lightning ckpts similarly to training. You can extract the `nn.Module` if you want to drop the library and use pure pytorch. This should be straightforward, check pytorch lightning documentation about inference.

The pretrained models are in `src/reference/`:  
For `DeepCharuco` you can find `longrun-epoch=99-step=369700.ckpt`
<details>
  <summary>Training details + plots</summary>
  
  Training time: `27 hours on GTX1080ti`  
  batch size: `32`  
  lr: `5e-3`  
  negative probability (in [transformations.py](src/transformations.py)): 0.05  
  Training plots:
  ![train_res](https://i.imgur.com/PFTL10P.png)
  ![train_res2](https://i.imgur.com/pdrC5C4.png)
  
  Where `l2_pixels` is the euclidean distance in pixels of the corners the model found during validation and
  `match_ratio` is the percentage of corners found over the total in each image. Please look at [metrics.py](models/metrics.py),
  they are not perfect metrics but provide useful insights of the model training.
</details>

For `RefineNet` you can find `second-refinenet-epoch-100-step=373k.ckpt`
<details>
  <summary>Training details + plots</summary>
  
  Training time: `22 hours on GTX1080ti`  
  total: `8` which means we take 8 corners from each single sample image for training ([train_refinenet.py](src/train_refinenet.py))  
  batch size: `256` (virtually `32` because `batch_size // total` is used)  
  lr: `1e-4`  
  negative probability (in `transformations.py`): 0.05  
  Training plots:
  ![train_res](https://i.imgur.com/5ddmaEB.png)
  ![train_res2](https://i.imgur.com/7kLH046.png)
  
  Where `val_dist_refinenet_pixels` is the euclidean distance in pixels of the predicted corner in `8x` resolution (so if dist_pixels is `3` the error in original resolution is `3/8` of a pixel.
  Please look at [metrics.py](models/metrics.py) for details.
</details>

---

If you setup the val data you can run `inference.py` for a preview of results on validation data used in training. To do inference on your images check `inference.py` code and adapt it to your needs, the idea is the following:
```python
import cv2
from inference import infer_image, load_models

# Load models
deepc_path = './reference/longrun-epoch=99-step=369700.ckpt'
refinenet_path = './reference/second-refinenet-epoch-100-step=373k.ckpt'
n_ids = 16  # The number of corners (models pretrained use 16 for default board)
deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids)

# Run inference on BGR image
img = cv2.imread('reference/samples_test/IMG_7412.png')

# The out_img will have corners plotted on it if draw_pred is True
# The keypoints format is (x, y, id_corner)
keypoints, out_img = infer_image(img, n_ids, deepc, refinenet, draw_pred=True)
```

## Training
Setup `config.yaml` and run `train.py` and `train_refinenet.py` to train each network separately.

## Common issues
I do not have the intention to create a package out of this so I had to use some workarounds with imports. In the current state `numba` caching might break causing an error like `ModuleNotFoundError: No module named 'model_utils'`. You just need to delete `src/__pycache__ and src/models/__pycache__` and everything will work fine.  

Feel free to open pull requests and issues if you have problems.
