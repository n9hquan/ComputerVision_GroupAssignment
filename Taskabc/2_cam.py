import cv2
import numpy as np

capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

width, height = 640, 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

sideBySide = True

mouseX, mouseY = width, height // 2

def draw_overlay(image, mouseX, mouseY):
    image = cv2.line(image, (mouseX, mouseY), (image.shape[1], mouseY), (0, 0, 255), 2)
    image = cv2.circle(image, (mouseX, mouseY), 5, (255, 255, 128), -1)
    return image

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Failed to grab frames.")
        break

    if sideBySide:
        combined = np.hstack((frameL, frameR))
    else:
        combined = cv2.addWeighted(frameL, 0.5, frameR, 0.5, 0)

    combined = draw_overlay(combined, mouseX, mouseY)

    cv2.imshow("Stereo View", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        sideBySide = not sideBySide

capL.release()
capR.release()
cv2.destroyAllWindows()
