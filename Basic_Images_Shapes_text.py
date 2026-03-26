import cv2
import numpy as np


img = np.zeros((512,512,3), np.uint8)

img[:] = 255, 0,0

# draw line on an image
#cv2.line(img,(0,0),(100,100),(0,255,0),2) # variable , starting point (width, height) , end point, color,  thickness of line
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)


# draw rectangle or square
cv2.rectangle(img, (350,100),(500,250),(0,255,0),cv2.FILLED)

# draw circle
cv2.circle(img, (150,400),50,(0,255,0),cv2.FILLED) # variable, start, diameter, color, thickness or filled

# write text

cv2.putText(img, "Muhammad Hammad Shah", (30,100), cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,255), 3) # variable, text in strings "", originated or staring point, font style, font scale, color, thickness

cv2.imshow('img',img)
cv2.waitKey(0)