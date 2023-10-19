from __future__ import print_function
import cv2
import random
import math
import copy
import scipy.io
import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from config import *
from utils import rotate_and_crop
#========================================================================#

class NightCC(data.Dataset):
    def __init__(self,train=True,folds_num=0):
        # self.root = 'E:/AdaVision/NightCC/Datasets/' # ./data/ColorCheckerRe/
        self.root = '/home/yangkf/GPnet/data/NightCC/'
        list_path = self.root  +'./imlist.txt'
        with open(list_path,'r') as f:
            self.all_data_list = f.readlines()

        self.data_list = [] 

        # if train:
        #     self.data_list = self.all_data_list
        # else:
        #     self.data_list = self.all_data_list
    
        folds = scipy.io.loadmat( self.root  +'./folds.mat')   
        if train:
            img_idx = folds['tr_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-149])
        else:
            img_idx = folds['te_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-149])  

        self.gts = scipy.io.loadmat( self.root  +'./gt.mat')  

        self.train = train                           
        
    def __getitem__(self,index):
        model = self.data_list[index]
        illums = []
        # filename
        fn = model.strip().split(' ')[0]
        img = cv2.imread(self.root  +'./img/'+fn,-1)
        illums = self.gts['gt'][int(fn[0:-4])-149]

        img = np.array(img,dtype='float32')
        illums = np.array(illums,dtype='float32')
       
        # ==============================================#

        mask = cv2.imread(self.root  +'./msk/'+ fn,cv2.IMREAD_GRAYSCALE)
        ret,mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)


        msk = 255-mask
        msk = np.array(msk,dtype='float32')
        msk[msk<1]=0
        msk[msk>=1]=1
        msk = np.expand_dims(msk,axis=2)
        msk = np.concatenate((msk,msk,msk),axis=-1)
        img = img*msk  # 20220127
        # ==============================================# 

        if self.train:
            img, illums = self.augment_train(img, illums)
            img = img * (1.0 / 16384)     
            img = img[:,:,::-1] # BGR to RGB       
            # ==============================================# 
            img = np.power(img,(1.0/2.2)) 
            img = img.transpose(2,0,1) # hwc to chw  
            img = torch.from_numpy(img.copy())
            img = img.type(torch.FloatTensor) 
 
            # ==============================================# 
        else:
            img = self.crop_test(img)
            img = img * (1.0 / 16384)            
            img = img[:,:,::-1] # BGR to RGB
            # ==============================================#
            img = np.power(img,(1.0/2.2)) 
            img = img.transpose(2,0,1) # hwc to chw  
            img = torch.from_numpy(img.copy())
            img = img.type(torch.FloatTensor) 
            # ==============================================# 

        illums = torch.from_numpy(illums.copy())
  
        return img,illums,fn
    
    def augment_train(self,ldr, illum):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        # angle = 0
        # s = FCN_INPUT_SIZE

        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)        
        flip_lr = random.randint(0, 1) # Left-right flip? 


        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR 
        
        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (FCN_INPUT_SIZE, FCN_INPUT_SIZE))
            if flip_lr:
                img = img[:, ::-1]

            img = img.astype(np.float32)

            # ==============================================# 
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 255)
            new_illum = np.clip(new_illum, 0.01, 100)

            return new_image, new_illum[::-1]
     
        return crop(ldr, illum)

    def crop_test(self,img,scale=0.5):
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img   
   
        #==============================================# 

    def __len__(self):
        return(len(self.data_list))

# ==============================================# 
from utils import *

if __name__=='__main__':
    
    dataset = NightCC(train=True,folds_num=0)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)    
    for ep in range(1):
        # time1 = time.time()
        for i, data in enumerate(dataload):
            if i>1:
                break  
            rgbimg,illums, fn = data
            print(rgbimg.shape)
            print(illums.shape)

#========================================================================================================#
            # b,c,w,h = rgbimg.shape
            # ww = np.int(np.floor(w/4))
            # hh = np.int(np.floor(h/4))
            # corp_img = torch.zeros((4,4))
            # corp_msk = torch.zeros((4,4))
            # for ii in range(4):
            #     for j in range(4):
            #         if ii==0 and j==0:
            #             corp_img = rgbimg[:,:,ww*j:ww*(j+1),hh*ii:hh*(ii+1)]
            #             corp_msk = msk[:,ww*j:ww*(j+1),hh*ii:hh*(ii+1)]
            #         else:
            #             corp_img = torch.cat((corp_img, rgbimg[:,:,ww*j:ww*(j+1),hh*ii:hh*(ii+1)]),0)
            #             corp_msk = torch.cat((corp_msk, msk[:,ww*j:ww*(j+1),hh*ii:hh*(ii+1)]),0)

            # print(corp_img[0:8,:,:,:].shape)
            # print(corp_img[8:16,:,:,:].shape)
            # print(corp_msk.shape)

            # for ii in range(4):
            #     if ii==0:
            #        rgbimg = torch.cat((corp_img[0,:,:,:],corp_img[1,:,:,:],corp_img[2,:,:,:],corp_img[3,:,:,:]),1)
            #        msk = torch.cat((corp_msk[0,:,:],corp_msk[1,:,:],corp_msk[2,:,:],corp_msk[3,:,:]),0)
            #     else:
            #         for j in range(4):
            #             if j==0:
            #                 rgbimg0 = corp_img[4*ii,:,:,:]
            #                 msk0 = corp_msk[4*ii,:,:]
            #             else:
            #                 rgbimg0 = torch.cat((rgbimg0,corp_img[4*ii+j,:,:,:]),1)
            #                 msk0 = torch.cat((msk0,corp_msk[4*ii+j,:,:]),0)

            #         rgbimg = torch.cat((rgbimg,rgbimg0),2)
            #         msk = torch.cat((msk,msk0),1)
#========================================================================================================#

            rgbimg = rgbimg.squeeze()
            rgbimg=torch.Tensor.cpu(rgbimg).detach().numpy()
            rgbimg= rgbimg.transpose(1,2,0)
            plt.matshow(rgbimg)
            plt.show()

