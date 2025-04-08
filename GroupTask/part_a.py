import cv2

# Initialize left and right camera
capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set resolution (optional)
width, height = 640, 480
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create VideoWriters
outL = cv2.VideoWriter('left.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
outR = cv2.VideoWriter('right.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

print("Recording... Press 'q' to stop.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    
    if retL and retR:
        outL.write(frameL)
        outR.write(frameR)
        cv2.imshow('Left', frameL)
        cv2.imshow('Right', frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
outL.release()
outR.release()
cv2.destroyAllWindows()
