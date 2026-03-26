import cv2

'''
In CV the x-axis is on the correct position but
In CV the y-axis is upside down meaning the y-axis is positive downside

'''

path = "Resources/lena.png"

img = cv2.imread(path)
print(img.shape) # check images shape ( width, height , color )

width, height = 1000, 1000

imgResize = cv2.resize(img,(width,height)) # change image width and height
print(imgResize.shape) # check resized image shape

imgCropped = img[200:300, 200:300] # cut or crop the image [y,x]
imgCropped_resized = cv2.resize(imgCropped,(img.shape[1],img.shape[0])) # 1 is for width and 0 is for height

cv2.imshow("Lena", img)
cv2.imshow("Resized", imgResize)
cv2.imshow("Cropped", imgCropped)
cv2.imshow("Cropped Resized", imgCropped_resized)


cv2.waitKey(0)