
import numpy as np
import torch
import torch.nn as nn
import cv2

base=cv2.imread('G:/1/high/1.png')
I=cv2.imread('G:/1/high/1.png')


l2_loss=nn.MSELoss()
I_shape=I.shape()
H=I_shape[2]
W =I_shape[3]
I=torch.from_numpy(I).float()
I=I.permute(2,0,1)
I=I.unsqueeze_(0)
I_B=I[:,0,:,:]
I_G=I[:,1,:,:]
I_R=I[:,2,:,:]
I_B=I_B.unsqueeze_(1)
I_G=I_G.unsqueeze_(1)
I_R=I_R.unsqueeze_(1)

base=torch.from_numpy(base).float()
base=base.permute(2,0,1)
base=base.unsqueeze_(0)
base_B=base[:,0,:,:]
base_G=base[:,1,:,:]
base_R=base[:,2,:,:]
base_B=base_B.unsqueeze_(1)
base_G=base_G.unsqueeze_(1)
base_R=base_R.unsqueeze_(1)

gradient_weights=torch.Tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]])
gradient_conv=torch.nn.Conv2d(1, 1, 3, stride=1)
gradient_conv.weight=nn.Parameter(gradient_weights,requires_grad=False)
base_B_gradient=gradient_conv(base_B)
base_G_gradient=gradient_conv(base_G)
base_R_gradient=gradient_conv(base_R)

Gauss_weights=torch.Tensor([[[[1,-2,1],[1,-2,1],[1,-2,1]]]])
Gauss_conv=torch.nn.Conv2d(1, 1, 3, stride=1)
Gauss_conv.weight=nn.Parameter(Gauss_weights,requires_grad=False)
Gauss_I_B=Gauss_conv(I_B)
Gauss_I_G=Gauss_conv(I_G)
Gauss_I_R=Gauss_conv(I_R)

sigma_B=pow(2*np.pi,0.5)/(6*(W-2)*(H-2))*(torch.sum(Gauss_I_B))
sigma_G=pow(2*np.pi,0.5)/(6*(W-2)*(H-2))*(torch.sum(Gauss_I_G))
sigma_R=pow(2*np.pi,0.5)/(6*(W-2)*(H-2))*(torch.sum(Gauss_I_R))

lamba_B=abs(2*sigma_B)
lamba_G=abs(2*sigma_G)
lamba_R=abs(2*sigma_R)

loss_B=l2_loss(base_B,I_B)+lamba_B*abs(torch.sum(base_B_gradient)/(H*W))
loss_G=l2_loss(base_G,I_G)+lamba_G*abs(torch.sum(base_G_gradient)/(H*W))
loss_R=l2_loss(base_R,I_R)+lamba_R*abs(torch.sum(base_R_gradient)/(H*W))

loss=loss_B+lamba_G+loss_R
print(loss)
