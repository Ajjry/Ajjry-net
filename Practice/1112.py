import cv2
import psnr
import ssim
import numpy as np
for i in  range(11):
    gt=cv2.imread("G:/SONY_GT_V/%d.jpg"%(int(i+1)))
    img=cv2.imread("G:/asd/%d.jpg"%(int(i+1)))

    psnr_data = psnr.PSNR(gt, img)  # psnr指标
    ssim_data = ssim.SSIM(gt ,img )  # ssim指标
    print(" psnr:" + str(psnr_data) + " ssim:"+str(ssim_data))