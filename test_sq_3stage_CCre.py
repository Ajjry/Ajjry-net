from __future__ import print_function
import os
import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

sys.path.append('./auxiliary/')
from model import squeezenet1_1,CreateNet,CreateNet_3stage
#from dataset  import *
from dataset_CCre import * # 20220423
from utils import *

torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--lrate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--pth_path0', type=str)
parser.add_argument('--pth_path1', type=str)
parser.add_argument('--pth_path2', type=str)
parser.add_argument('--alpha1', default=0.33, type=float,help='alpha1')
parser.add_argument('--alpha2', default=0.33, type=float,help='alpha2')
opt = parser.parse_args()
print (opt)

val_loss = AverageMeter()
errors = []

#create network
network = CreateNet_3stage().cuda()
network.eval()
for i in range(3):
    dataset_test = ColorCheckerRE(train=False,folds_num=i)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,shuffle=False, num_workers=opt.workers)
    len_dataset_test = len(dataset_test)
    print('Len_fold:',len(dataset_test))
    if i == 0:
        pth_path = opt.pth_path0
    elif i == 1:      
        pth_path = opt.pth_path1
    elif i == 2:
        pth_path = opt.pth_path2    
    #load parameters
    network.load_state_dict(torch.load(pth_path))
    for i,data in enumerate(dataloader_test):
        img, label,fn = data
        img = Variable(img.cuda())
        label = Variable(label.cuda())
        pred1,pred2,pred3, = network(img)
        loss = get_angular_loss(torch.mul(torch.mul(pred1,pred2),pred3),label)
        val_loss.update(loss.item())
        errors.append(loss.item())
        print('Model: %s, AE: %f'%(fn[0],loss.item())) 

mean,median,trimean,bst25,wst25,pct95 = evaluate(errors)
print('Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f'%(mean,median,trimean,bst25,wst25,pct95))













