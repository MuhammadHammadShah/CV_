import cv2
import numpy as np

# horizontal stacking in numpy, vertical stack also

img1 = cv2.imread("Resources/lena.png")
img2 = img1.copy()
img3 = cv2.imread("Resources/lena.png",0)


print(img1.shape)
print(img2.shape)
print(img3.shape)

img1 = cv2.resize(img1, (0,0), None, 0.5, 0.5)
img2 = cv2.resize(img2, (0,0), None, 0.5, 0.5)
img3 = cv2.resize(img3, (0,0), None, 0.5, 0.5)

"""
If We Remove the Below line of code the image will remain in Gray color and the shape will be (512,512) which cannot be stacked will (512,512,3) so it will throw an error:
ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 3 dimension(s) and the array at index 2 has 2 dimension(s)
"""

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

hor = np.hstack((img1,img2,img3))
ver = np.vstack((img1,img2,img3))

cv2.imshow("Horizontal", hor)
cv2.imshow("Vertical", ver)
cv2.waitKey(0)

