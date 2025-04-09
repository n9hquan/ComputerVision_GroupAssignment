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

# StereoBM settings
numDisparities = 16 * 12  # Must be divisible by 16
blockSize = 5  # Must be odd

# stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

stereo = cv2.StereoSGBM_create(
    numDisparities=16*12,  # multiple of 16
    blockSize=5,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Camera read failed")
        break

    # Convert to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)

    # Compute disparity map
    disparity = stereo.compute(grayL, grayR)

    # Normalize the disparity for visualization
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # Apply a colormap to better visualize depth
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # Show images
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Disparity (Depth Map)", disp_color)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
