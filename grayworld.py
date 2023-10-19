import numpy as np
def grayworld(demosaing_img):
    RGBmean = demosaing_img.mean(axis=1).mean(axis=0)
    meangray=RGBmean.mean(axis=0)
    demosaing_img[:, :, 2] = demosaing_img[:, :, 2].astype(np.float16) * (meangray / RGBmean[2])
    demosaing_img[:, :, 1] = demosaing_img[:, :, 1].astype(np.float16) * (meangray / RGBmean[1])
    demosaing_img[:, :, 0] = demosaing_img[:, :, 0].astype(np.float16) * (meangray / RGBmean[0])
    demosaing_img=np.minimum(demosaing_img,16383)
    demosaing_img=demosaing_img.astype(np.uint16)
    return demosaing_img