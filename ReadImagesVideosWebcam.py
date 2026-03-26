import cv2

img = cv2.imread("Resources/lena.png") # This function reads the images

cv2.imshow("Lena", img) # This function show the images

cv2.waitKey(1000) # Wait for 1 sec before closing > if we only write 0 it will wait infinity