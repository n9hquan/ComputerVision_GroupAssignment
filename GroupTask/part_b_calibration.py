import cv2
import numpy as np
import glob
import os

chessboard_size = (8, 6)
square_size = 1.0  # real-world square size (e.g., in cm)
output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

# === PREPARE OBJECT POINTS ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

left_images = sorted(glob.glob(r'GroupTask\left_images\left_2.jpg'))
right_images = sorted(glob.glob(r'GroupTask\right_images\right_2.jpg'))

print(f"Found {len(left_images)} left-right image pairs.")

for left_path, right_path in zip(left_images, right_images):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
        cv2.imwrite(os.path.join(output_dir, f"cornersL_{os.path.basename(left_path)}"), imgL)
        cv2.imwrite(os.path.join(output_dir, f"cornersR_{os.path.basename(right_path)}"), imgR)

print("Finished collecting corner points.")

# === CALIBRATE INDIVIDUAL CAMERAS ===
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# === STEREO CALIBRATION ===
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags
)

# === RECTIFICATION ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0
)

import json
camera_params = {
    "camera_matrix_left": mtxL.tolist(),
    "dist_coeff_left": distL.tolist(),
    "camera_matrix_right": mtxR.tolist(),
    "dist_coeff_right": distR.tolist(),
    "rotation_matrix": R.tolist(),
    "translation_vector": T.tolist(),
    "essential_matrix": E.tolist(),
    "fundamental_matrix": F.tolist(),
    "Q": Q.tolist()  # ← ADD THIS!
}

with open(os.path.join(output_dir, "camera_params_PA.json"), "w") as f:
    json.dump(camera_params, f, indent=4)

print("Calibration data with Q matrix saved to camera_params_PA.json")



map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)

# === PROCESS ONE SAMPLE PAIR ===
sample_left = cv2.imread(left_images[0])
sample_right = cv2.imread(right_images[0])

rectL = cv2.remap(sample_left, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(sample_right, map2x, map2y, cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(output_dir, 'rectified_left.png'), rectL)
cv2.imwrite(os.path.join(output_dir, 'rectified_right.png'), rectR)

# === COMPUTE DISPARITY ===
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# === SHOW & SAVE DISPARITY MAP ===
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)
cv2.imwrite(os.path.join(output_dir, "disparity_map.png"), disp_norm)

# === DEPTH CALCULATION ===
disparity[disparity <= 0.1] = 0.1  # Prevent division by zero
focal_length = P1[0, 0]
baseline = abs(T[0])
depth_map = (focal_length * baseline) / disparity

np.save(os.path.join(output_dir, "depth_map.npy"), depth_map)
print("Depth map and disparity saved in 'output/'")
