import cv2
import os
img = cv2.imread("G:/z/DSC_0133.JPG")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("G:/z/10.jpg" , hsv[:,:,2])