import numpy as np
import cv2
import os
import glob

# Set the dimensions of the chessboard pattern (in interior corners)
CHESSBOARD_SIZE = (9, 6)

# Define the termination criteria for the iterative calibration algorithm
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the path to the directory containing the input images
IMAGE_DIR = '../data_demo/calib_frames/'

# Load the input images from the specified directory
image_files = sorted(glob.glob(IMAGE_DIR + '*.png'))
images = [cv2.imread(f) for f in image_files]

# Initialize the arrays to store the object points and image points for calibration
object_points = []  # 3D points of the corners in the chessboard pattern
image_points = []   # 2D points of the corners in the input images

# Prepare the object points for the chessboard pattern
object_point = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
object_point[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Find the chessboard corners in each input image
for i, image in enumerate(images[::5]):  # Here I take only a few images..
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        object_points.append(object_point)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
        image_points.append(corners_refined)

        if "DISPLAY" in os.environ:
            cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners_refined, found)
            cv2.destroyAllWindows()
            cv2.imshow(f'Images {i+1}', image)
            cv2.waitKey(1)

# Perform camera calibration using the object and image points
print('Running calibration...')
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, images[0].shape[0:2][::-1], None, None)

# Compute and display the reprojection error
mean_error = 0
for i in range(len(object_points)):
    image_points_est, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, distortion_coeffs)
    error = cv2.norm(image_points[i], image_points_est, cv2.NORM_L2) / len(image_points_est)
    mean_error += error
print(f"Mean reprojection error: {mean_error/len(object_points)}")

# Save the camera matrix and distortion coefficients to a file in the same directory as the input images
output_file = os.path.join(IMAGE_DIR, 'camera_params.npz')
np.savez(output_file, camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
print(f"Camera matrix and distortion coefficients saved to {output_file}")

