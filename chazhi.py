import numpy as np
import cv2
def get_green(img,h,w):
        a = [_ % 2 for _ in range(0, w)]
        b = a + [_ % 2 for _ in range(1, w + 1)]
        c = np.tile(np.array(b), (int((h + 1) / 2), 1))
        c = np.resize(c, (h, w))
        c=c.astype(np.uint8)
        img=img*c
        return img

def get_red(img, h, w):
        a = np.array(([1, 0] * int(w / 2) + ([1] * (w % 2))))
        a = np.expand_dims(a, axis=0)
        b = np.array(([1, 0] * int(h / 2) + ([1] * (h % 2))))
        b = np.expand_dims(b, axis=1)
        b=b@a
        b = b.astype(np.uint8)
        img = img * b
        return img
def get_blue(img,h,w):
        a = np.array(([0, 1] * int(w / 2) + ([1] * (w % 2))))
        a = np.expand_dims(a, axis=0)
        b = np.array(([0, 1] * int(h / 2) + ([1] * (h % 2))))
        b = np.expand_dims(b, axis=1)
        b=b@a
        b = b.astype(np.uint8)
        img = img * b
        return img
def chazhi(rawimg,h,w):
        rawimg = np.expand_dims(rawimg, axis=2)
        rawimg = np.concatenate((rawimg, rawimg, rawimg), axis=2)
        splitimg = cv2.split(rawimg)
        splitimg[2] = get_red(splitimg[2], h, w)
        splitimg[1] = get_green(splitimg[1], h, w)
        splitimg[0] = get_blue(splitimg[0], h, w)
        byer_img = cv2.merge(splitimg)
        return splitimg