

import numpy
import math
from skimage.measure.simple_metrics import compare_psnr
def PSNR(x_image, y_image, max_value=255.0):
    return compare_psnr(x_image, y_image, max_value)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
