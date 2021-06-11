
import torch
from torch.nn import *

def Conv_1(in_dim, out_dim, stride = 1):
    return Sequential(
        Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
        BatchNorm2d(out_dim),
        LeakyReLU(negative_slope=0.1, inplace=True)
    )

def Conv_2(in_dim, out_dim, stride):
    return Sequential(
        Conv2d(in_dim, in_dim, 3, stride, 1, groups=in_dim, bias=False),
        BatchNorm2d(in_dim),
        LeakyReLU(negative_slope= 0.1,inplace=True),

        Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
        BatchNorm2d(out_dim),
        LeakyReLU(negative_slope= 0.1,inplace=True),
    )

class MobileNetV1(Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = Sequential(
            Conv_1(3, 8, 2),  
            Conv_2(8, 16, 1),  
            Conv_2(16, 32, 2),  
            Conv_2(32, 32, 1),
            Conv_2(32, 64, 2), 
            Conv_2(64, 64, 1),  
        )
        self.stage2 = Sequential(
            Conv_2(64, 128, 2), 
            Conv_2(128, 128, 1),
            Conv_2(128, 128, 1), 
            Conv_2(128, 128, 1), 
            Conv_2(128, 128, 1), 
            Conv_2(128, 128, 1),
        )
        self.stage3 = Sequential(
            Conv_2(128, 256, 2), 
            Conv_2(256, 256, 1), 
        )
        self.avg = AdaptiveAvgPool2d((1,1))
        self.fc = Linear(256, 1000)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return [x1,x2,x3]  
      
      
      
mbnet=MobileNetV1()
      
