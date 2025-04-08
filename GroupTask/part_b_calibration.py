import cv2 as cv
import numpy as np
import glob
import json
import os

# Chessboard dimensions (corners inside the board)
CHESSBOARD_SIZE = (8, 6)  # columns, rows
SQUARE_SIZE = 40  # millimeters (change based on your printed chessboard)

# Criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) * square_size ...
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

left_images = sorted(glob.glob("snapshot_left_1.jpg"))
right_images = sorted(glob.glob("snapshot_right_1.jpg"))

for left_path, right_path in zip(left_images, right_images):
    imgL = cv.imread(left_path)
    imgR = cv.imread(right_path)

    if imgL is None or imgR is None:
        print(f"Could not load image pair: {left_path}, {right_path}")
        continue

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    retR, cornersR = cv.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)

        cornersL2 = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR2 = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

        imgpoints_left.append(cornersL2)
        imgpoints_right.append(cornersR2)
    else:
        print(f"Chessboard not found in pair: {left_path}, {right_path}")

if not objpoints:
    raise ValueError("No valid pairs with chessboard corners found.")

# Calibrate each camera separately
_, mtxL, distL, _, _ = cv.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
_, mtxR, distR, _, _ = cv.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=flags
)

# Save parameters to JSON
params = {
    "camera_matrix_left": mtxL.tolist(),
    "dist_coeff_left": distL.tolist(),
    "camera_matrix_right": mtxR.tolist(),
    "dist_coeff_right": distR.tolist(),
    "rotation_matrix": R.tolist(),
    "translation_vector": T.tolist(),
    "essential_matrix": E.tolist(),
    "fundamental_matrix": F.tolist()
}

with open("camera_params.json", "w") as f:
    json.dump(params, f, indent=4)

print("Stereo calibration complete. Parameters saved to camera_params.json")
