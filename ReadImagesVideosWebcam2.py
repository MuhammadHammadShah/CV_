import cv2

frameWidth = 640  # frame's width captured by camera
frameHeight = 480 # frame's height captured by camera

cap = cv2.VideoCapture(0) # this captures the frames from video
                          # 0 is for device camera or the path can be specified for the video or camera in strings

# cap.set(3, frameWidth)  # the 3 refers to width, this cap.set sets the width for the frame to be show
# cap.set(4, frameHeight)  # the 4 refers to height, this cap.set sets the height for the frame to be show

while True:
    success, img = cap.read() # cap.read returns two thing a boolean True, False to declare the frame loaded successfully stored in "success" and the frame in "img"

    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): # this make the video going on until exited
        break