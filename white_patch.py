
def whitepatch(denoise_img,RGBmax):
    denoise_img[:,:,2] = denoise_img[:,:,2] * RGBmax[1] / RGBmax[2]
    denoise_img[:,:,0] = denoise_img[:,:,0] * RGBmax[1] / RGBmax[0]

    '''
    for i in range(h):
        for j in range(w):
            splitimage[2][i, j] = splitimage[2][i, j] * RGBmax[1] / RGBmax[2]
    for i in range(h):
        for j in range(w):
            splitimage[0][i, j] = splitimage[0][i, j] * RGBmax[1] / RGBmax[0]
    '''
    return denoise_img
