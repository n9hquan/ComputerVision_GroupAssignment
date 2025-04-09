import cv2
import numpy as np
import json

# === Load camera parameters from calibration ===
with open("output/camera_params_PA.json", "r") as f:
    params = json.load(f)

cameraMatrix1 = np.array(params["camera_matrix_left"])
distCoeffs1 = np.array(params["dist_coeff_left"])
cameraMatrix2 = np.array(params["camera_matrix_right"])
distCoeffs2 = np.array(params["dist_coeff_right"])
R = np.array(params["rotation_matrix"])
T = np.array(params["translation_vector"])

# === Setup ===
width, height = 640, 480
image_size = (width, height)

# === Rectification ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    image_size, R, T, alpha=0
)

# === Rectification Maps ===
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2)

# === Open USB Cameras ===
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# === Stereo Matching Algorithm ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=7,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

print("Press 'q' to exit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Failed to grab frames")
        break

    # === Rectify Frames ===
    rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    # === Grayscale Conversion ===
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # === Optional Smoothing ===
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)

    # === Compute Disparity Map ===
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # === Normalize for Visualization ===
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # === Show Results ===
    cv2.imshow("Left", rectL)
    cv2.imshow("Right", rectR)
    cv2.imshow("Disparity", disp_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
