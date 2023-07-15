import cv2
import numpy as np
import os

root_dir = "E:/workspace/pycharm_workspace/BME/DATA/Normal/"

def calcAndDrawHist(image, color, mask):
    hist = cv2.calcHist([image], [0], mask, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


def hough_circle_demo(image):
    # 霍夫圆检测对噪声敏感，边缘检测消噪
    h, w, s = image.shape
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 60, param1=65, param2=40, minRadius=50, maxRadius=0)
    circles = np.uint16(np.around(circles))  #把circles包含的圆心和半径的值变成整数
    img = np.zeros((h, w), dtype='uint8')
    for i in circles[0, :]:
        # cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
        cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), -1)
        cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255), -1)
    # cv2.imshow("circle image", image)
    cv2.imshow("img_mask", img)
    return img


if __name__ == '__main__':
    image = cv2.imread(os.path.join(root_dir + 'ASC-L-20160521.jpg'))
    b, g, r = cv2.split(image)

    mask = hough_circle_demo(image)
    print("mask", mask.shape)

    histImgB = calcAndDrawHist(b, [255, 0, 0], mask)
    histImgG = calcAndDrawHist(g, [0, 255, 0], mask)
    histImgR = calcAndDrawHist(r, [0, 0, 255], mask)
    histImg = calcAndDrawHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), [125, 125, 125], mask)

    cv2.imshow("histImgB", histImgB)
    cv2.imshow("histImgG", histImgG)
    cv2.imshow("histImgR", histImgR)
    cv2.imshow("histImg", histImg)
    cv2.imshow("Img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
