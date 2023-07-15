import cv2
import greypixel
img = cv2.imread("F:/eyes/DATA/Normal/ASC-R-20160521.jpg")
img1=cv2.imread("F:/1.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img=img-img1
cv2.imshow("1", img)
cv2.waitKey(0)