import torch
import numpy as np
import cv2
from models.model_utils import pre_bgr_image, pred_to_keypoints

try:
    import torch_tensorrt
except ModuleNotFoundError:
    print("NO TRT SUPPORT")

def pred_to_keypoints(loc_hat: torch.Tensor, ids_hat: torch.Tensor, dust_bin_ids: int):
    ids = torch.argmax(ids_hat, dim=1)
    loc = torch.argmax(loc_hat, dim=1)
    ids = torch.where(loc == 64, dust_bin_ids, ids)
    mask = ids != dust_bin_ids
    indices = torch.nonzero(mask, as_tuple=False)
    ids_found = ids[mask]
    region_pixel = loc[mask]
    xs = 8 * indices[:, -1] + (region_pixel % 8)
    ys = 8 * indices[:, -2] + (region_pixel // 8).to(torch.int)
    return torch.cat((xs.unsqueeze(1), ys.unsqueeze(1)), dim=1), ids_found, indices[:, 0]

import torch.nn.functional as F

def extract_patches(images: torch.Tensor, keypoints: torch.Tensor, indices: torch.Tensor, patch_size: int = 24) -> torch.Tensor:
    # Get the number of images
    num_images = images.shape[0]

    # Pad the images with zeros
    padding = patch_size // 2
    padded_images = F.pad(images.squeeze(1), (padding, padding, padding, padding), mode='constant', value=0)

    # Extract the patches centered around the keypoints for each image
    patches = []
    for i in range(num_images):
        img_indices = indices == i
        padded_img = padded_images[i]
        img_keypoints = keypoints[img_indices]
        ys = img_keypoints[:, 1, None] + torch.arange(patch_size, device=padded_img.device)
        p1 = torch.index_select(padded_img, 0, ys.view(-1,)).view(img_keypoints.shape[0], -1, padded_img.shape[-1])
        xs = (img_keypoints[:, 0, None] + torch.arange(patch_size, device=padded_img.device)).unsqueeze(1)
        image_patches = torch.gather(p1, 2, xs.expand(-1, p1.size(1), -1))
        patches.append(image_patches)
    return torch.vstack(patches)


def infer_image_batched(images: np.ndarray, dust_bin_ids: int, deepc, refinenet):
    images_gray = [pre_bgr_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
    images_gray = torch.tensor(np.array(images_gray), device='cuda')
    loc_hat, ids_hat = deepc.forward(images_gray)
    keypoints, ids_found, indices = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)
    patches = extract_patches(images_gray, keypoints, indices)

    padding = 1024 - patches.size(0)  # TODO
    if padding > 0:
        patches = torch.cat((patches, torch.zeros(padding, *patches.shape[1:], device='cuda')), dim=0)
        keypoints = torch.cat((keypoints, torch.zeros(padding, keypoints.shape[1], device='cuda')), dim=0)
    keypoints = refinenet.forward(patches, keypoints.to(torch.float))
    keypoints = keypoints[:-padding]

    keypoints = keypoints.cpu().numpy()
    ids_found = ids_found.cpu().numpy()
    indices = indices.cpu().numpy()

    kpts = []
    for i in range(images_gray.shape[0]):
        mask = indices == i
        k_i = keypoints[mask]
        ids_i = ids_found[mask]
        kpts.append(np.array([[k[0], k[1], idx] for k, idx in sorted(zip(k_i, ids_i),
                                                                 key=lambda x: x[1])]))
    return kpts


if __name__ == '__main__':
    deepc_path = "./reference/deepc_trt.ts"
    refinenet_path = "./reference/refinenet.ts"
    deepc = torch.jit.load(deepc_path)
    refinenet = torch.jit.load(refinenet_path)

    # Inference test on custom image
    BS = 64
    SAMPLE_IMAGE = './reference/samples_test/IMG_7412.png'
    img = cv2.imread(SAMPLE_IMAGE)
    img = np.repeat(img[None], BS, axis=0)

    n_ids = 16

    # Warmup
    for i in range(5):
        keypoints = infer_image_batched(img, n_ids, deepc, refinenet)

    # Run inference
    import time
    n = 500
    t = time.time()
    for i in range(n):
        keypoints = infer_image_batched(img, n_ids, deepc, refinenet)
    print(f"\033[95m--->FPS: {(n * BS)/(time.time() - t):0.1f} \033[0m")