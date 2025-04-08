import cv2
import numpy as np
import json

# === Load stereo images ===
imgL = cv2.imread('snapshot_left_1.jpg')
imgR = cv2.imread('snapshot_right_1.jpg')

# === Load camera parameters from JSON ===
with open('camera_params.json') as f:
    params = json.load(f)

K1 = np.array(params['camera_matrix_left'])
D1 = np.array(params['dist_coeff_left'])[0]
K2 = np.array(params['camera_matrix_right'])
D2 = np.array(params['dist_coeff_right'])[0]
R = np.array(params['rotation_matrix'])
T = np.array(params['translation_vector'])

image_size = imgL.shape[:2][::-1]  # (width, height)

# === Stereo rectification ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

# === Undistort and rectify maps ===
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# === Convert to grayscale ===
grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

# === Compute disparity map ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # must be divisible by 16
    blockSize=7,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# === Reproject to 3D space ===
points_3D = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2RGB)
mask = disparity > disparity.min()

out_points = points_3D[mask]
out_colors = colors[mask]

# === Save to .ply ===
def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_colors = np.hstack([verts, colors])
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts_colors)))
        np.savetxt(f, verts_colors, fmt='%f %f %f %d %d %d')

