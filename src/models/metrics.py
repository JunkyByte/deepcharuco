import torch
from typing import Optional
from torchmetrics import Metric


def pred_argmax(loc_hat: torch.Tensor, ids_hat: torch.Tensor, dust_bin_ids: int):
    assert loc_hat.ndim == 4 and ids_hat.ndim == 4
    # (N, C, H/8, W/8)
    ids_argmax = torch.argmax(ids_hat, dim=1)
    loc_argmax = torch.argmax(loc_hat, dim=1)

    # Mask ids_hat using loc_hat dust_bin
    # This way we will do an argmax only over best ids with valid location
    ids_argmax[loc_argmax == 64] = dust_bin_ids
    return loc_argmax, ids_argmax


def pred_to_keypoints(loc_hat: torch.Tensor, ids_hat: torch.Tensor, dust_bin_ids: int):
    assert loc_hat.ndim == 4 and ids_hat.ndim == 4
    loc_argmax, ids_argmax = pred_argmax(loc_hat, ids_hat, dust_bin_ids)
    kps, ids = label_to_keypoints(loc_argmax, ids_argmax, dust_bin_ids)
    return kps, ids


def label_to_keypoints(loc: torch.Tensor, ids: torch.Tensor, dust_bin_ids: int):
    assert loc.ndim == 3 and ids.ndim == 3
    mask = ids != dust_bin_ids
    roi = torch.argwhere(mask)
    ids_found = ids[mask]
    region_pixel = loc[mask]

    # Recover exact pixel in original resolution
    xs = 8 * roi[:, -1] + (region_pixel % 8)
    ys = 8 * roi[:, -2] + torch.div(region_pixel, 8, rounding_mode='floor')
    return torch.cat((xs.unsqueeze(1), ys.unsqueeze(1)), dim=1).float(), ids_found


class DC_Metrics(Metric):
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, dust_bin_ids):
        super().__init__()
        self.add_state("distance", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("ratio", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.px_margin = 3
        self.dust_bin_ids = dust_bin_ids

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        (loc_x, ids_x), (loc_target, ids_target) = preds, target

        bs = loc_x.shape[0]
        l2_sum = 0.
        ratio_sum = 0.
        atleast = False
        for i in range(bs):
            l2_dist = None
            keypoint, id = pred_to_keypoints(loc_x[i].unsqueeze(0),
                                             ids_x[i].unsqueeze(0),
                                             self.dust_bin_ids)
            keypoint_target, id_target = label_to_keypoints(loc_target[i].unsqueeze(0),
                                                            ids_target[i].unsqueeze(0),
                                                            self.dust_bin_ids)
            l2_dist = self.compute_l2_distance(keypoint, id, keypoint_target, id_target)
            ratio = self.compute_ratio(keypoint, id, keypoint_target, id_target)

            if l2_dist is not None:
                atleast = True
                l2_sum += l2_dist
                ratio_sum += ratio

        if atleast:
            self.distance += l2_sum / bs
            self.ratio += ratio_sum / bs

    def compute_ratio(self, keypoints, ids, target_keypoints, target_ids):  # TODO merge the initial part of these two compute
        matches = torch.zeros((len(target_ids),))

        if matches.numel() == 0:
            return None

        # Loop over each unique id in target_ids
        for i, id in enumerate(torch.unique(target_ids)):
            # Find the indices of keypoints and target_keypoints with the same id
            mask = (ids == id).nonzero().squeeze(1)
            target_mask = (target_ids == id).nonzero().squeeze(1)

            # If there are no matching keypoints, skip to the next id
            if mask.numel() == 0 or target_mask.numel() == 0:
                continue

            # Compute the L2 distance between the matching keypoints and target_keypoints
            dist = torch.cdist(keypoints[mask], target_keypoints[target_mask], p=2).squeeze(1)
            max_dist, _ = torch.max(dist, dim=0)

            # Store the maximum distance (i.e. worse) for this id and see if under min
            if max_dist < self.px_margin:
                matches[i] = 1
        return matches.mean()

    def compute_l2_distance(self, keypoints, ids, target_keypoints, target_ids):
        # Initialize an empty tensor to store the distances
        distances = torch.zeros((len(target_ids),))

        if distances.numel() == 0:
            return None

        # Loop over each unique id in target_ids
        found = 0
        for i, id in enumerate(torch.unique(target_ids)):
            # Find the indices of keypoints and target_keypoints with the same id
            mask = (ids == id).nonzero().squeeze(1)
            target_mask = (target_ids == id).nonzero().squeeze(1)

            # If there are no matching keypoints, skip to the next id
            if mask.numel() == 0 or target_mask.numel() == 0:
                continue

            # Compute the L2 distance between the matching keypoints and target_keypoints
            dist = torch.cdist(keypoints[mask], target_keypoints[target_mask], p=2).squeeze(1)
            max_dist, _ = torch.max(dist, dim=0)

            # Store the maximum distance for this id
            distances[i] = max_dist
            found += 1

        return distances.sum() / max(1, found)

    def compute(self):
        return self.distance, self.ratio


class Refinenet_Metrics(Metric):
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("distance", default=torch.tensor(0.), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loc_x, loc_target = preds, target
        loc_x = loc_x.squeeze(1)

        bs = loc_x.shape[0]
        d = 64
        m = loc_x.view(bs, -1).argmax(1)
        loc_indices = torch.cat((torch.div(m, d, rounding_mode='floor').view(-1, 1),
                                (m % d).view(-1, 1)), dim=1).unsqueeze(1)

        m = loc_target.view(bs, -1).argmax(1)
        target_indices = torch.cat((torch.div(m, d, rounding_mode='floor').view(-1, 1),
                                   (m % d).view(-1, 1)), dim=1).unsqueeze(1)

        dist = torch.cdist(loc_indices.float(), target_indices.float(), p=2).squeeze(1)
        self.distance += dist.mean()

    def compute(self):
        return self.distance


if __name__ == '__main__':
    import numpy as np
    import sys
    sys.path.append('./')

    import configs
    from configs import load_configuration
    from data import CharucoDataset

    config = load_configuration(configs.CONFIG_PATH)
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    sample = next(iter(dataset_val))
    image, label = sample.values()
    loc, ids = label

    dc_metrics = DC_Metrics(dust_bin_ids=16)

    loc_x = torch.tensor(np.random.randint(low=0, high=65, size=(32, 65, 30, 40))).float()
    ids_x = torch.tensor(np.random.randint(low=0, high=17, size=(32, 16, 30, 40))).float()

    loc_target = torch.tensor(loc[None, ...])
    ids_target = torch.tensor(ids[None, ...])

    loc_target = torch.repeat_interleave(loc_target, 32, dim=0)
    ids_target = torch.repeat_interleave(ids_target, 32, dim=0)

    print(dc_metrics((loc_x, ids_x), (loc_target, ids_target)))

    rn_metrics = Refinenet_Metrics()

    loc_x = torch.tensor(np.random.randint(low=0, high=100000, size=(8, 1, 64, 64))).float()
    loc_x /= loc_x.max()

    loc_target = torch.tensor(np.random.randint(low=0, high=100000, size=(8, 64, 64))).float()
    loc_target /= loc_target.max()

    print(rn_metrics(loc_x, loc_target))
