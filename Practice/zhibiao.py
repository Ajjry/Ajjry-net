import cv2
import psnr
import ssim
import numpy as np
psnr1=0.
ssim1=0.
for i in range(1,16):

    gt=cv2.imread("G:/1/high/%d.png"%(int(i)))
    img=cv2.imread("G:/fusion/4/%d.jpg"%(int(i)))
    #gt=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img =  cv2.resize(img,(6016,4016), interpolation=cv2.INTER_AREA)

    psnr_data = psnr.PSNR(gt, img)  # psnr指标
    psnr1+=psnr_data
    ssim_data = ssim.SSIM(gt ,img )  # ssim指标

    ssim1+=ssim_data
    print(i,":"," psnr:" + str(psnr_data) + " ssim:"+str(ssim_data))
print("psnr_avg:",psnr1/15,"ssim_avg",ssim1/15)