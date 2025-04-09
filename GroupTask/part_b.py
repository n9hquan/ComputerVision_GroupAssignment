import cv2
import time

# Initialize left and right camera
capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set resolution
width, height = 640, 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if cameras opened successfully
if not capL.isOpened() or not capR.isOpened():
    print("❌ Error: Could not open one or both cameras.")
    exit()

print(" Showing camera feeds. Will take snapshots after 5 seconds...")

start_time = time.time()
snapshot_taken = False

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if retL and retR:
        cv2.imshow('Left', frameL)
        cv2.imshow('Right', frameR)

        # After 5 seconds, take snapshots
        if not snapshot_taken and time.time() - start_time >= 5:
            cv2.imwrite('snapshot_left_PA_book.jpg', frameL)
            cv2.imwrite('snapshot_right_PA_book.jpg', frameR)
            print(" Snapshots saved as 'snapshot_left.jpg' and 'snapshot_right.jpg'")
            snapshot_taken = True

    else:
        print("⚠️ Failed to grab frames.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Quit by user.")
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
