import cv2
import numpy as np

path = "Resources/lena.png"

img = cv2.imread(path) # if we write (path,0) , due to 0 parameter the frame will convert into grayscale frame from RGB

kernel = np.ones((5,5), np.uint8)  # 5x5 matrix


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (9,9),0)
imgCanny = cv2.Canny(imgBlur,100,200) # the two values are treshold, more threshol less edge detection
imgDilated = cv2.dilate(imgCanny,kernel, iterations=2) # more iteration more thickness
imgEroded = cv2.erode(imgDilated, kernel, iterations=2)  # more iteration less thickness

# kernel will always be in odd number, the more the kernel number the more the blurness, the 2nd value is for sigma.
# cv2.imshow("Lena", img)
cv2.imshow("BLur_lena", imgBlur)
cv2.imshow("Gray_Lena", imgGray)
cv2.imshow("Canny_lena", imgCanny)
cv2.imshow("Dilated_lena", imgDilated)
cv2.imshow("Eroded_lena", imgEroded)
cv2.waitKey(0)
