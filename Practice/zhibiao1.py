import cv2
import psnr
import ssim

img=cv2.imread("G:/ceshi/LOL_DE_99_2/mynet14.jpg")
gt=cv2.imread("G:/1/high/1.png")
#gt=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img =  cv2.resize(img,(6016,4016), interpolation=cv2.INTER_AREA)
psnr_data = psnr.PSNR(gt, img)  # psnr指标
ssim_data = ssim.SSIM(gt ,img )  # ssim指标
print(" psnr:" + str(psnr_data) + " ssim:"+str(ssim_data))