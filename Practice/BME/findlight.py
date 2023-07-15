import numpy as np
import argparse
import cv2
import os
def find_high_light(path,r):

    def image_crop(image,rect):
        y1 = np.maximum(rect[0][1],rect[1][1])
        y2 = np.minimum(rect[2][1],rect[3][1])
        x1 = np.maximum(rect[0][0], rect[3][0])
        x2 = np.minimum(rect[1][0], rect[2][0])
        crop_image=image[int(y1):int(y2),int(x1):int(x2),:]

        return crop_image
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=path,help = "path to the image file")
    ap.add_argument("-r", "--radius", type = int,default=r, help = "radius of Gaussian blur; must be odd")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image_shape=image.shape
    mask=np.ones(image_shape)
    mask = mask[:, :, 0]*255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    cnt = np.array([[maxLoc[0]-args["radius"],maxLoc[1]-args["radius"]],
                    [maxLoc[0]+args["radius"],maxLoc[1]-args["radius"]],
                    [maxLoc[0]+args["radius"],maxLoc[1]+args["radius"]],
                    [maxLoc[0]-args["radius"],maxLoc[1]+args["radius"]]])
    x1=maxLoc[0]-args["radius"]
    x2=maxLoc[0]+args["radius"]
    y1=maxLoc[1]-args["radius"]
    y2=maxLoc[1]+args["radius"]
    mask[y1:y2,x1:x2]=0

    img1=image_crop(image,cnt)

    return img1,mask
name=os.listdir("F:\\eyes\\test\\")
name.sort(key=lambda x: int(x[:-4]))
for i in range(3000):
    path=os.path.join("F:\\eyes\\test\\", name[i])
    r=129
    img1,mask=find_high_light(path,r)
    cv2.imwrite("F:\\eyes\\test_mask\\%d.jpg" % (int(i+1)), mask)

