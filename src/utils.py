import cv2
import os
import numpy as np


def compute_l2_distance(keypoints, ids, target_keypoints, target_ids):
    # Initialize an empty array to store the distances
    distances = np.zeros((len(target_ids),))

    if distances.size == 0:
        return None

    # Loop over each unique id in target_ids
    for i, id in enumerate(np.unique(target_ids)):
        # Find the indices of keypoints and target_keypoints with the same id
        mask = np.nonzero(ids == id)[0]
        target_mask = np.nonzero(target_ids == id)[0]

        # If there are no matching keypoints, skip to the next id
        if mask.size == 0 or target_mask.size == 0:
            continue

        # Compute the L2 distance between the matching keypoints and target_keypoints
        dist = np.linalg.norm(keypoints[mask] - target_keypoints[target_mask], ord=2, axis=1)
        max_dist = np.max(dist)

        # Store the maximum distance for this id
        distances[i] = max_dist

    return distances


def pixel_error(kpts_raw, kpts_ref, kpts_target):
    if not set(kpts_raw[:, 2]).issubset(set(kpts_target[:, 2])):
        return None, None
    d = compute_l2_distance(kpts_raw[:, :2], kpts_raw[:, 2],
                            kpts_target[:, :2], kpts_target[:, 2])
    d_ref = compute_l2_distance(kpts_ref[:, :2], kpts_ref[:, 2],
                                kpts_target[:, :2], kpts_target[:, 2])
    d_raw_ref = compute_l2_distance(kpts_ref[:, :2], kpts_ref[:, 2],
                                    kpts_raw[:, :2], kpts_raw[:, 2])

    found = np.unique(kpts_raw[:, 2])
    print(f'Errors in pixels of the {len(found)}/{len(kpts_target[:, 2])} kpts found:')
    # for ith, id in enumerate(kpts_target[:, 2]):
    #     if id not in found:
    #         continue
    #     print(f'Marker {id:<2} Error Raw: {d[ith]:<5.3f} Error Refinet: {d_ref[ith]:<5.3f}')
    print(f'Mean error raw: {d.mean():<5.3f} Max error raw: {d.max():<5.3f}')
    print(f'Mean error ref: {d_ref.mean():<5.3f} Max error ref: {d_ref.max():<5.3f}')
    print(f'Mean dist raw/ref: {d_raw_ref.mean():<5.3f} Max dist raw/ref: {d_raw_ref.max():<5.3f}')
    return d.mean(), d_ref.mean()


def save_video(frames, output_path, fps=30):
    # Get the shape of the first frame to determine the resolution of the video
    height, width, _ = frames[0].shape

    # Create a video writer object with the specified filename, codec, framerate, and resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        writer.write(frame)

    # Release the video writer object and print a message
    writer.release()
    print(f"Saved video to {os.path.abspath(output_path)}")
