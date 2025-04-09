import cv2
import numpy as np

# Open the left and right cameras
capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

width, height = 640, 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# StereoSGBM settings
numDisparities = 16 * 12
blockSize = 5

stereo = cv2.StereoSGBM_create(
    numDisparities=numDisparities,
    blockSize=blockSize,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

focal_length = 578.6  # in pixels
baseline = 0.08       # in meters

point_x = width // 2
point_y = height // 2

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Camera read failed")
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    grayL = cv2.medianBlur(grayL, 5)
    grayR = cv2.medianBlur(grayR, 5)

    # Compute disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Get disparity at chosen point
    disp_value = disparity[point_y, point_x]

    # Estimate depth
    if disp_value > 0:
        depth = (focal_length * baseline) / disp_value
    else:
        depth = float('inf')  # Invalid disparity

    # Normalize disparity for visualization
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # Draw point on depth map
    cv2.circle(disp_color, (point_x, point_y), 6, (255, 255, 255), 2)

    # Display depth info
    if depth < 0.5:
        text = "WARNING: Object < 50cm!"
        color = (0, 0, 255)
    else:
        text = f"Distance: {depth:.2f} m"
        color = (0, 255, 0)

    cv2.putText(disp_color, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Show results
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Disparity (Depth Map)", disp_color)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()