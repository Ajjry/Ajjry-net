import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import  Retinex_function_kernel
class Encoderblock(nn.Module):
    def __init__(self, input_nf, output_nf):
        super(Encoderblock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nf, output_nf, kernel_size=3, stride=1, padding=0, bias=True),

            nn.MaxPool2d(2, stride=2, ceil_mode=False),
            nn.BatchNorm2d(output_nf),
            nn.ReLU(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UPblock(nn.Module):
    def __init__(self, input_nf, output_nf):
        super(UPblock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nf, output_nf, kernel_size=3, stride=1, padding=0, bias=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(output_nf),
            nn.ReLU(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class enhance_net(nn.Module):


    def __init__(self,chanel,kernel_size,device):
        super(enhance_net, self).__init__()
        gama = torch.linspace(0.5, 0.5, 16)
        bias = torch.linspace(1e-4, 1e-4, 16)
        n = torch.linspace(0.5,8.0,16)

        number_f = 32
        self.device = device
        self.gama = nn.Parameter(torch.FloatTensor(gama), True)
        self.bias = nn.Parameter(torch.FloatTensor(bias), False)
        self.n = nn.Parameter(torch.FloatTensor(n), True)
        self.chanel = chanel
        kernel = torch.ones([1,kernel_size, kernel_size]) / kernel_size ** 2
        self.kernel = nn.Parameter(torch.FloatTensor(kernel),True)
        self.kernel_size = kernel_size
        self.Retinex = Retinex_function_kernel(self.n,self.gama, self.bias,self.kernel, self.kernel_size,chanel)

        self.down1 = Encoderblock(16*chanel , number_f)
        self.down2 = Encoderblock(number_f, number_f * 2)
        self.down3 = Encoderblock(number_f * 2, number_f * 4)

        self.trans = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(number_f * 4, number_f * 4, kernel_size = 3, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(number_f * 4),
            nn.ReLU(),
        )

        self.up1 = UPblock(number_f * 8, number_f * 2)
        self.up2 = UPblock(number_f * 4, number_f)
        self.up3 = UPblock(number_f * 2, number_f)

        self.out = nn.Sequential(
            nn.Conv2d(number_f, chanel, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        #x_mean=torch.mean(x,dim=1,keepdim=True)

        x_shape = x.shape
        b, c, h, w = x_shape
        x_fusion = torch.zeros((b, c * 16, h, w))

        x_fusion, out_local, x_filter = self.Retinex(x, x_fusion)

        #x_fusion=torch.cat([r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16],dim=1)
        x_fusion = x_fusion.to(self.device)
        x1 = self.down1(x_fusion)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.trans(x3)
        x5 = self.up1(torch.cat([x4, x3], 1))
        x6 = self.up2(torch.cat([x5, x2], 1))
        x7 = self.up3(torch.cat([x6, x1], 1))
        # print(torch.max(x7))
        x_out = torch.sigmoid(self.out(x7))
        if self.chanel == 3:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x_out1 = torch.mean(x_out,dim=1,keepdim=True)
            x_result = x_out1 * (x / x_mean)
        else:
            x_result = x_out

        return x_result,self.n ,self.bias ,self.gama, self.kernel, out_local, x_filter
