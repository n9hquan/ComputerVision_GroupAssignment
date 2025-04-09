import numpy as np
import cv2
import open3d as o3d
import json
import os

# === CONFIGURATION ===
output_dir = 'output/'
depth_map_path = os.path.join(output_dir, 'depth_map.npy')
image_path = os.path.join(output_dir, 'rectified_left.png')
camera_params_path = os.path.join(output_dir, 'camera_params.json')

# === LOAD DATA ===
depth_map = np.load(depth_map_path)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Open3D expects RGB

with open(camera_params_path, 'r') as f:
    params = json.load(f)

Q = np.array(params["Q"]) if "Q" in params else None
if Q is None:
    raise ValueError("Q matrix not found in camera_params.json. Add this line after stereoRectify: params['Q'] = Q.tolist()")

# === REPROJECT TO 3D ===
h, w = depth_map.shape
points = []
colors = []

for y in range(h):
    for x in range(w):
        Z = depth_map[y, x]
        if Z <= 0 or Z > 10000:  # Skip invalid or far points
            continue
        X = (x - Q[0, 3]) / Q[0, 0] * Z
        Y = (y - Q[1, 3]) / Q[1, 1] * Z
        points.append([X, Y, Z])
        colors.append(image[y, x] / 255.0)

# === CREATE POINT CLOUD ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# === VISUALIZE ===
o3d.visualization.draw_geometries([pcd])
