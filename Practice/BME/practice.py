import cv2
import psnr
import ssim
import demo


max_index1=[]
for j in range(12):
    out1 = []
    for i in range(1):
        imgstr="F:/eyes/349/%d/output_%d.png"%(int(j),int(i))
        gtstr="F:/eyes/349/%d/input.png"%(int(j))
        img=cv2.imread(imgstr)
        gt=cv2.imread(gtstr)

        img=demo.demo(imgstr,gtstr)
        #gt=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img =  cv2.resize(img,(6016,4016), interpolation=cv2.INTER_AREA)
        cv2.imwrite("F:/eyes/349/%d/%d.png"%(int(j),int(i)),img)
        psnr_data = psnr.PSNR(gt, img)  # psnr指标
        ssim_data = ssim.SSIM(gt ,img )
        out=psnr_data+20*ssim_data# ssim指标
        #print(" psnr:" + str(psnr_data) + " ssim:"+str(ssim_data))

        out1.append(out)

    #print(out1)
    max_index = out1.index(max(out1))
    print(max_index)
    max_index1.append(max_index)
print(max_index1)