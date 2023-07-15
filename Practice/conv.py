import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

img=cv2.imread("G:/1/high/1.png")
# 第一一个卷积层，我们可以看到它的权值是随机初始化的
w=torch.nn.Conv2d(3,3,3,padding=1)
print(w.weight)


# 第一种方法
print("1.使用另一个Conv层的权值")
q=torch.nn.Conv2d(3,3,3,stride=1) # 假设q代表一个训练好的卷积层
print(q.weight.shape) # 可以看到q的权重和w是不同的
w.weight=q.weight # 把一个Conv层的权重赋值给另一个Conv层
print(w.weight)

# 第二种方法
print("2.使用来自Tensor的权值")
ones=torch.Tensor([[[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]]],[[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]]],[[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]],[[1,-2,1],[1,-2,1],[1,-2,1]]]]) # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
w.weight=torch.nn.Parameter(ones) # 把Tensor的值作为权值赋值给Conv层，这里需要先转为torch.nn.Parameter类型，否则将报错
print(w.weight)
img=torch.from_numpy(img).float()
img=img.permute(2,0,1)
img=img.unsqueeze_(0)
a=w(img)
img.squeeze_(0)
img=img.permute(1,2,0)
img=img.numpy()
img = img.astype(np.uint8)
cv2.imshow('1',img)
cv2.waitKey(0)