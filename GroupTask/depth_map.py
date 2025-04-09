import numpy as np
import cv2

# Load the depth map
depth_map = np.load('output/depth_map.npy')

# Normalize for visualization
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = np.uint8(depth_vis)

# Show image
cv2.imshow("Depth Map", depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
