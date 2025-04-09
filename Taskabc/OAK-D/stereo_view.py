import cv2
import numpy as np

# === Camera Setup ===
capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Left camera
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Right camera

width, height = 640, 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# === Display toggle setup ===
cv2.namedWindow("Stereo Pair")
sideBySide = True

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Failed to grab frames")
        break

    # Resize to match in case sizes differ slightly
    frameR = cv2.resize(frameR, (frameL.shape[1], frameL.shape[0]))

    if sideBySide:
        # Show side by side
        imOut = np.hstack((frameL, frameR))
    else:
        # Blend the two images for comparison
        imOut = cv2.addWeighted(frameL, 0.5, frameR, 0.5, 0)

    cv2.imshow("Stereo Pair", imOut)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        sideBySide = not sideBySide

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
