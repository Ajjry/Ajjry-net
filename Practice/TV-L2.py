import cv2
import numpy as np

def TV_L2(S,nlevel,ww,hh):
    # S = (img.astype('float32'))/255
    betamax = 1e5
    # ww,hh = img.shape
    ########################otfFx########################
    psfFx = np.zeros((1,hh-2))
    psfFx = np.column_stack((-1,psfFx))
    psfFx = np.column_stack((psfFx,1))
    psfFxAdd = np.zeros((ww-1,hh))
    psfFx = np.row_stack((psfFx,psfFxAdd))
    otfFx = abs(np.fft.fft2(psfFx))
    ########################otfFy########################
    psfFy = np.zeros((ww-2,1))
    psfFy = np.row_stack((-1,psfFy))
    psfFy = np.row_stack((psfFy,1))
    psfFyAdd = np.zeros((ww,hh-1))
    psfFy = np.column_stack((psfFy,psfFyAdd))
    otfFy = abs(np.fft.fft2(psfFy))
    ###############################################################
    Normin1 = np.fft.fft2(S)
    Denormin2 = np.zeros((ww,hh))
    Denormin2 = Denormin2.astype('float32')
    Denormin2 = abs(otfFx)**2 + abs(otfFy)**2
    beta = 2*nlevel
    while beta < betamax:
        level = nlevel / beta
        Denormin = 1 + beta * Denormin2
        # h - v subproblem
        S_1th_column = np.diff(S)
        S_1toend_column = S[:,0]-S[:,hh-1]
        u = np.column_stack((S_1th_column,S_1toend_column))
        u = np.maximum(np.abs(u)-level,0)*np.sign(u)
        S_1th_row = np.diff(S,axis=0)
        S_1toend_row = S[0]-S[ww-1]
        v = np.row_stack((S_1th_row,S_1toend_row))
        v = np.maximum(np.abs(v)-level,0)*np.sign(v)
        Normin2 = np.column_stack(((u[:,hh-1]-u[:,0]),-np.diff(u)))
        Normin2 = Normin2 + np.row_stack(((v[ww-1]-v[0]),-np.diff(v,axis=0)))
        FS = (Normin1 + beta * np.fft.fft2(Normin2)) / Denormin
        S = np.real(np.fft.ifft2(FS))
        beta = beta * 2
    S = 255*((S-S.min())/(S.max()-S.min()))
    S = S.astype('uint8')
    return(S)

def decomp(img):
    ww,hh = img.shape
    Ilum = (img.astype('float32'))/255
    nlevel_1 = 2e-2
    Ibase_1 = TV_L2(Ilum,nlevel_1,ww,hh)
    ret,dst = cv2.threshold(Ibase_1,0,255,cv2.THRESH_OTSU)
    index_background = dst<1
    background = img[index_background]
    standard = np.std(background)
    nlevel_2 = (standard**(4/5))/1000;
    Ibase_2 = TV_L2(Ilum,nlevel_2,ww,hh)
    nlevel_3 = nlevel_2+0.5
    Ibase_3 = TV_L2(Ilum,nlevel_3,ww,hh)
    return(dst,Ibase_2,Ibase_3)




img = cv2.imread('G:/z/3.jpg',0)
BW,Ibase_2,Ibase3 = decomp(img)
cv2.imshow('img',BW)
cv2.waitKey(0)
cv2.destroyAllWindows()

