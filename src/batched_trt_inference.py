import torch
import numpy as np
import time
import cv2
import torch.multiprocessing as mp
from models.model_utils import pre_bgr_image, pred_to_keypoints
import queue

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


def worker_deepc(queue_images, dust_bin_ids, out_queue: mp.Queue):
    s_deepc = torch.cuda.Stream()
    deepc_path = "./reference/deepc_trt.ts"
    deepc = torch.jit.load(deepc_path)
    while True:
        with torch.cuda.stream(s_deepc):
            torch.cuda.nvtx.range_push("Worker_deepc_start")

            # t0 = time.time()
            images = queue_images.get()
            # print(f'deepc waited for: {time.time() - t0:0.2f}')
            if images is StopIteration: # TODO
                break

            images_gray = [pre_bgr_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
            images_gray = torch.tensor(np.array(images_gray), device='cuda')
            loc_hat, ids_hat = deepc.forward(images_gray)
            keypoints, ids_found, indices = pred_to_keypoints(loc_hat, ids_hat, dust_bin_ids)
            patches = extract_patches(images_gray, keypoints, indices)

            # patches.share_memory_()
            # keypoints.share_memory_()
            # ids_found.share_memory_()
            # indices.share_memory_()

            out_queue.put((patches, keypoints, ids_found, indices))
            torch.cuda.nvtx.range_pop()

def worker_ref(deepc_queue: mp.Queue, out_queue: mp.Queue):
    s_ref = torch.cuda.Stream()
    refinenet_path = "./reference/refinenet.ts"
    refinenet = torch.jit.load(refinenet_path)
    while True:
        with torch.cuda.stream(s_ref):
            torch.cuda.nvtx.range_push("Worker_ref_start")

            # t0 = time.time()
            out = deepc_queue.get()
            # print(f'ref waited for: {time.time() - t0:0.2f}')

            if out is StopIteration: # TODO
                break

            patches, keypoints, ids_found, indices = out
            actual_size = patches.size(0)
            padding = 16 * BS - actual_size  # TODO  16*64 = 1024
            if padding > 0:
                patches = torch.cat((patches, torch.zeros(padding, *patches.shape[1:], device='cuda')), dim=0)
                kpts_ref = torch.cat((keypoints, torch.zeros(padding, keypoints.shape[1], device='cuda')), dim=0)
            kpts_ref = refinenet.forward(patches, kpts_ref.to(torch.float))
            kpts_ref = kpts_ref[:-padding]

            # TODO:
            ids_found_clone = ids_found.clone()
            indices_clone = indices.clone()
            del keypoints, ids_found, indices

            torch.cuda.nvtx.range_pop()
            out_queue.put((kpts_ref, ids_found_clone, indices_clone, actual_size))


def worker_remaining(ref_queue: mp.Queue, out_queue: mp.Queue):
    s_rem = torch.cuda.Stream()
    while True:
        with torch.cuda.stream(s_rem):
            torch.cuda.nvtx.range_push("Worker_remaining_start")

            # t0 = time.time()
            out = ref_queue.get()
            # print(f'remaining waited for: {time.time() - t0:0.2f}')

            if out is StopIteration: # TODO
                break

            keypoints_gpu, ids_found_gpu, indices_gpu, size = out

            # keypoints = keypoints_gpu.cpu().numpy()
            keypoints = keypoints_gpu.to('cpu', non_blocking=True).numpy() # TODO? SYNC?
            del keypoints_gpu

            # ids_found = ids_found_gpu.cpu().numpy()
            ids_found = ids_found_gpu.to('cpu', non_blocking=True).numpy() # TODO? SYNC?
            del ids_found_gpu

            # indices = indices_gpu.cpu().numpy()
            indices = indices_gpu.to('cpu', non_blocking=True).numpy() # TODO? SYNC?
            del indices_gpu

            torch.cuda.nvtx.range_pop()

            kpts = []
            for i in range(size):
                mask = indices == i
                k_i = keypoints[mask]
                ids_i = ids_found[mask]
                kpts.append(np.array([[k[0], k[1], idx] for k, idx in sorted(zip(k_i, ids_i),
                                                                            key=lambda x: x[1])]))
            out_queue.put(kpts)


def parallel_runner(dust_bin_ids):
    # Parallel inference setup!
    processes = []
    torch.multiprocessing.set_start_method('spawn', force=True)

    main_to_deepc = mp.Queue()
    deepc_to_ref = mp.Queue()
    ref_to_rem = mp.Queue()
    rem_to_main = mp.Queue()
    queues = [main_to_deepc, deepc_to_ref, ref_to_rem, rem_to_main]

    processes = []
    processes.append(mp.Process(target=worker_deepc, args=(
        main_to_deepc, dust_bin_ids, deepc_to_ref)))
    processes.append(mp.Process(target=worker_ref, args=(
        deepc_to_ref, ref_to_rem)))
    processes.append(mp.Process(target=worker_remaining, args=(
        ref_to_rem, rem_to_main)))

    for process in processes:
        process.start()
    print('Started processes...')

    # for process in processes:
    #     process.join()
    
    return ParallelRunnerWrap(processes, queues, main_to_deepc, rem_to_main)

class ParallelRunnerWrap:
    def __init__(self, processes, queues, q_push: mp.Queue, q_get: mp.Queue):
        self.processes = processes
        self.queues = queues
        self._q_push = q_push
        self._q_get = q_get
    
    def send_batch(self, images):
        # print("Pushing image!")
        self._q_push.put(images)
        return None
    
    def get(self, timeout=None):
        try:
            # print('Trying to get!')
            return self._q_get.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        for q in reversed(self.queues):
            q.put(StopIteration)  # TODO This works but might be fragile (wait?)
        for process in self.processes:
            process.join()
        return None


BS = 64
if __name__ == '__main__':
    # Inference test on custom image
    SAMPLE_IMAGE = './reference/samples_test/IMG_7412.png'
    img = cv2.imread(SAMPLE_IMAGE)
    img = np.repeat(img[None], BS, axis=0)

    n_ids = 16

    # The parallel runner :)
    print("Runner is building...")
    runner = parallel_runner(n_ids)

    # Warmup
    print('Running warmup! Sending samples')
    for i in range(2):
        runner.send_batch(img)
    res_1 = runner.get()
    res_2 = runner.get()
    print('Runner is warm!')

    # Run inference
    import time
    n = 50
    t = time.time()
    for i in range(n):
        runner.send_batch(img)

    for i in range(n):
        keypoints = runner.get()
    print(f"\033[95m--->FPS: {(n * BS)/(time.time() - t):0.1f} \033[0m")

    runner.close()