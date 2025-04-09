import cv2
import numpy as np
import json
import os

# === CONFIGURATION ===
output_dir = 'output/'
left_image_path = 'left_image.jpg'
right_image_path = 'right_image.jpg'

# === LOAD CAMERA PARAMS ===
with open(os.path.join(output_dir, "camera_params.json"), "r") as f:
    params = json.load(f)

# Convert to numpy
mtxL = np.array(params["camera_matrix_left"])
distL = np.array(params["dist_coeff_left"])
mtxR = np.array(params["camera_matrix_right"])
distR = np.array(params["dist_coeff_right"])
R = np.array(params["rotation_matrix"])
T = np.array(params["translation_vector"])
image_size = cv2.imread(left_image_path).shape[1::-1]  # (width, height)

# === RECTIFICATION ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

# === LOAD AND RECTIFY IMAGES ===
imgL = cv2.imread(left_image_path)
imgR = cv2.imread(right_image_path)

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite(os.path.join(output_dir, "rectified_left.png"), rectL)
cv2.imwrite(os.path.join(output_dir, "rectified_right.png"), rectR)

# === COMPUTE DISPARITY ===
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# Save disparity image
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
cv2.imwrite(os.path.join(output_dir, "disparity_map.png"), disp_vis)

# === COMPUTE DEPTH MAP ===
disparity[disparity <= 0.1] = 0.1
focal_length = P1[0, 0]
baseline = abs(T[0][0])  # Make sure it's a scalar
depth_map = (focal_length * baseline) / disparity
np.save(os.path.join(output_dir, "depth_map.npy"), depth_map)

print("Disparity and depth map saved to 'output/'")
