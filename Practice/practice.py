import os
import cv2
import numpy as np
import torch
import torch.nn as nn
img=cv2.imread('G:/1/high/1.png')
img = (np.asarray(img) / 255.0)
img1=cv2.imread('G:/fusion/1/5.jpg')
img1= (np.asarray(img1) / 255.0)
img=torch.from_numpy(img).float()
img = img.permute(2, 0, 1)
img1=torch.from_numpy(img1).float()
img1 = img1.permute(2, 0, 1)
angle_color = nn.CosineSimilarity(dim=1, eps=1e-7)
color_loss=torch.mean(angle_color(img1,img))
print(10*color_loss)
