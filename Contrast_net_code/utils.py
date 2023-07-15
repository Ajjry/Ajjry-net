import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import numpy as np
import torch
from skimage.metrics import structural_similarity
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
import torchvision
import kornia.filters as filters
import scipy.stats as st
import scipy
import kornia
import math
import cv2
def SSIM(x_image, y_image, max_value=1.0, win_size=3, use_sample_covariance=True):
    x_image=x_image.permute(0,2,3,1)
    y_image = y_image.permute(0, 2, 3, 1)
    x = x_image.data.cpu().numpy().astype(np.float32)
    y = y_image.data.cpu().numpy().astype(np.float32)
    ssim=0
    for i in range(x.shape[0]):
        ssim += structural_similarity(x[i,:,:,:],y[i,:,:,:], win_size=win_size, data_range=max_value, multichannel=True)
    return (ssim/x.shape[0])

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def F_set(img, kernel_size, sigma, in_ch = 1, out_ch = 1):
    X = torch.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size).cuda()
    Y = torch.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size).cuda()
    x, y = torch.meshgrid(X, Y)
    kernel = torch.abs(x) + torch.abs(y)
    kernel = kernel.cuda()
    kernel = 1 / (((kernel / sigma) ** 2) + 1)
    kernel = kernel / torch.sum(kernel)

    kernel = kernel.expand(in_ch, out_ch, kernel_size, kernel_size)
    kernel = nn.Parameter(data=kernel, requires_grad=False)
    re = nn.ReflectionPad2d(kernel_size // 2).cuda()
    img = re(img)
    x1 = img[:, 0]
    x2 = img[:, 1]
    x3 = img[:, 2]
    x1 = F.conv2d(x1.unsqueeze(1), kernel)
    x2 = F.conv2d(x2.unsqueeze(1), kernel)
    x3 = F.conv2d(x3.unsqueeze(1), kernel)
    x = torch.cat([x1, x2, x3], dim=1)
    return  x

def Retinex(x, gama, bias, sigma, kernel_size):

    I = F_set(x, kernel_size, sigma)
    #print(x.shape,gauss.shape)
    x = x / (I + bias)
    r = x / ( x + gama )
    return r
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight *  (h_tv / count_h + w_tv / count_w) / batch_size
class L_TV_low(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV_low, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow(((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight *  (h_tv / count_h + w_tv / count_w) / batch_size
class kernel_loss(torch.nn.Module):
    def __init__(self, kernel_size):
        super(kernel_loss, self).__init__()
        self.kernel = nn.Conv2d(kernel_size = kernel_size, stride = 1, padding = 0, bias = False)
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.bias = nn.Parameter(torch.FloatTensor(1e-5), True)
        self.TV = L_TV()

    def forward(self, x):
        I = self.padding(x)
        I = self.kernel(I)
        x = I / ( x + self.bias)
        loss = self.TV(x)
        return loss

class Retinex_function(torch.nn.Module):
    def __init__(self,n , gama,bias,chanel):
        super(Retinex_function, self).__init__()
        #self.TV = L_TV()
        self.bias = bias
        self.gama = gama
        self.chanel = chanel
        self.n = n
    def forward(self, x, x_fusion):
        for i in range(16):
            r = 1 / (1 + torch.exp(- self.n[i] * (x - self.gama[i]))) + self.bias[i]
            # r = x / (x + self.gama[i])
            x_fusion[:, 0 + i * self.chanel : self.chanel * (1+ i):, :] = r
        return x_fusion


class Retinex_function_kernel(torch.nn.Module):
    def __init__(self ,n, gama, bias, kernel, kernel_size,chanel):
        super(Retinex_function_kernel, self).__init__()
        self.n = n
        self.chanel = chanel
        self.kernel = kernel
        self.padding = nn.ReflectionPad2d(kernel_size // 2).cuda()
        self.bias = bias
        #self.TV = L_TV()
        self.gama = gama
    def forward(self, x, x_fusion):
        for i in range(16):

            r = x / (x + self.gama[i] + self.bias[i])
            # r = torch.pow(x, self.n[i])/(torch.pow(x, self.n[i]) + torch.pow(self.gama[i], self.n[i])+self.bias[i])
            x_mean = filters.filter2d(r, self.kernel, border_type='replicate')

            out_local = r * x_mean    #11
            # out_local = r * (r/(x_mean+1e-4)) #22
            # r = (1 / (1 + torch.exp(- self.n[i] * (out_local - self.gama[i]))))+self.bias[i]


            x_fusion[:, 0 + i * self.chanel: (1 + i) * self.chanel, :, :] = out_local
        return x_fusion, out_local ,x_mean


def cal_center_loss(kernel,kernel_size):
    X = torch.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size).cuda()
    Y = torch.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size).cuda()
    x, y = torch.meshgrid(X, Y)
    weights = torch.abs(x) + torch.abs(y)
    return torch.sum(torch.abs(weights*kernel))/kernel_size**2

def cal_kernel_loss(kernel):
    sum_loss = 0.
    b = kernel.shape[0]
    for i in range(b):
        sum_loss += torch.abs(torch.sum(kernel[i])-1)
    #kernel_size = kernel.shape[-1]
    #center_loss = cal_center_loss(kernel,kernel_size)
    #sparse_loss = torch.sum(torch.pow(torch.abs(kernel),0.5))
    return 15 * sum_loss/b

def contrast_loss(x,gt):
    mean_img = kornia.filters.box_blur(x, [11, 11], border_type='replicate')
    mean_gt = kornia.filters.box_blur(gt, [11, 11], border_type='replicate')

    residual_img = kornia.filters.laplacian(x, 11, border_type='replicate')
    residual_gt = kornia.filters.laplacian(gt, 11, border_type='replicate')

    contrast_gt = (torch.abs(residual_gt)+0.0001) / ((mean_gt)+0.0001)
    contrast_img = (torch.abs(residual_img)+0.0001) / ((mean_img)+0.0001)

    contrast_loss = torch.mean(torch.abs(contrast_img - contrast_gt))

    return contrast_loss


def save_img(img,img_path):
    enhanced_image = torch.squeeze(img, 0)
    enhanced_image = enhanced_image.permute(1, 2, 0)
    enhanced_image = np.asarray(enhanced_image.cpu())
    enhanced_image = enhanced_image * 255.0
    cv2.imwrite(img_path, enhanced_image)

