import cv2
import numpy as np
import os




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
    gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,1000, param1=65, param2=40, minRadius=50, maxRadius=0)
    if circles is  None:
        ret1, thresh1 = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)
        thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 10, 1000, param1=65, param2=40, minRadius=50, maxRadius=0)
    #print(circles)
    circles = np.uint16(circles)  #把circles包含的圆心和半径的值变成整数
    img = np.zeros((h, w), dtype='uint8')
    for i in circles[0, :]:
        # cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
        cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), -1)
        cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255), -1)
    # cv2.imshow("circle image", image)
    #cv2.imshow("img_mask", img)
    return circles, img


def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect


def max_rect(img, mask):
    cnts,a = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = order_points(c.reshape(c.shape[0], 2))
    #print(rect)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.drawContours(mask, cnts, -1, (0, 0, 255), 3)
    mask = cv2.line(mask, (int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])), (0, 0, 0), 3)
    mask = cv2.line(mask, (int(rect[0][0]), int(rect[0][1])), (int(rect[3][0]), int(rect[3][1])), (0, 0, 0), 3)
    mask = cv2.line(mask, (int(rect[1][0]), int(rect[1][1])), (int(rect[2][0]), int(rect[2][1])), (0, 0, 0), 3)
    mask = cv2.line(mask, (int(rect[2][0]), int(rect[2][1])), (int(rect[3][0]), int(rect[3][1])), (0, 0, 0), 3)
    #cv2.imshow('img_mask', mask)

    img = cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    img = cv2.line(img, (int(rect[0][0]), int(rect[0][1])), (int(rect[1][0]), int(rect[1][1])), (0, 0, 0), 3)
    img = cv2.line(img, (int(rect[0][0]), int(rect[0][1])), (int(rect[3][0]), int(rect[3][1])), (0, 0, 0), 3)
    img = cv2.line(img, (int(rect[1][0]), int(rect[1][1])), (int(rect[2][0]), int(rect[2][1])), (0, 0, 0), 3)
    img = cv2.line(img, (int(rect[2][0]), int(rect[2][1])), (int(rect[3][0]), int(rect[3][1])), (0, 0, 0), 3)
    #cv2.imshow('img_rect', img)
    return rect


def image_crop(image,rect,mask):
    y1 = np.maximum(rect[0][1],rect[1][1])
    y2 = np.minimum(rect[2][1],rect[3][1])
    x1 = np.maximum(rect[0][0], rect[3][0])
    x2 = np.minimum(rect[1][0], rect[2][0])
    crop_image=image[int(y1):int(y2),int(x1):int(x2),:]
    crop_mask = mask[int(y1):int(y2), int(x1):int(x2), :]
    crop_image_shape=crop_image.shape
    h=crop_image_shape[0]
    w=crop_image_shape[1]
    img1 = crop_image[:int(h / 2), :int(w / 2), :]
    img2 = crop_image[int(h / 2):, :int(w / 2), :]
    img3 = crop_image[:int(h / 2), int(w / 2):, :]
    img4 = crop_image[int(h / 2):, int(w / 2):, :]
    mask1 =crop_mask[:int(h / 2), :int(w / 2), :]
    mask2= crop_mask[int(h / 2):, :int(w / 2), :]
    mask3= crop_mask[:int(h / 2), int(w / 2):, :]
    mask4= crop_mask[int(h / 2):, int(w / 2):, :]
    return img1,img2,img3,img4,mask1,mask2,mask3,mask4


if __name__ == '__main__':
    path="F:/eyes/data_R_L/normal/"
    img_path=os.path.join(path,'L/')
    mask_path = os.path.join(path, 'L_mask/')
    name = os.listdir(img_path)
    name.sort(key=lambda x: int(x[:-4]))
    name_mask = os.listdir(mask_path)
    name_mask.sort(key=lambda x: int(x[:-4]))
    for i in range(1000):
        img = cv2.imread(os.path.join(img_path, name[i]))
        #img = cv2.imread("F:/eyes/data_R_L/normal/L/3.jpg")
        cir, mask = hough_circle_demo(img)
        re = max_rect(img, mask)
        mask = cv2.imread(os.path.join(mask_path, name_mask[i]))
        img1,img2,img3,img4,mask1,mask2,mask3,mask4 = image_crop(img, re,mask)

        cv2.imwrite("F:/eyes/data_R_L/normal/L_crop/%d(1).jpg" % (int(i + 1)),img1)
        cv2.imwrite("F:/eyes/data_R_L/normal/L_crop/%d(2).jpg" % (int(i + 1)),img2)
        cv2.imwrite("F:/eyes/data_R_L/normal/L_crop/%d(3).jpg" % (int(i + 1)),img3)
        cv2.imwrite("F:/eyes/data_R_L/normal/L_crop/%d(4).jpg" % (int(i + 1)),img4)

        cv2.imwrite("F:/eyes/data_R_L/normal/mask_L_crop/%d(1).jpg" % (int(i + 1)), mask1)
        cv2.imwrite("F:/eyes/data_R_L/normal/mask_L_crop/%d(2).jpg" % (int(i + 1)), mask2)
        cv2.imwrite("F:/eyes/data_R_L/normal/mask_L_crop/%d(3).jpg" % (int(i + 1)), mask3)
        cv2.imwrite("F:/eyes/data_R_L/normal/mask_L_crop/%d(4).jpg" % (int(i + 1)), mask4)


