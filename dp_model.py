#**********************************************
# depth prediction model (from luma CTU to depth map)
#**********************************************
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform

class DP_Net(nn.Module): # depth prediction
    def __init__(self, out_channel):
        super(DP_Net, self).__init__()
        self.out_ch = int(out_channel) # 4 or 1

        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=1, padding=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(2,2))
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1,1))
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1,1))
		
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=(1,1))
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1,1))
        self.conv8 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=(1,1))
        self.conv9 = nn.Conv2d(8, self.out_ch, kernel_size=3, stride=1, padding=(1,1))


    def forward(self, x):
        x1 = f.relu(self.conv1(x)) #32*64*64
        x2 = f.max_pool2d(f.relu(self.conv2(x1)), 2) #64*32*32
        x3 = f.max_pool2d(f.relu(self.conv3(x2)), 2) #64*16*16
        x4 = f.max_pool2d(f.relu(self.conv4(x3)), 2) #32*8*8
        x5 = f.relu(self.conv5(x4)) #32*8*8
        # multi pooling + concat
        x5_1 = f.interpolate(f.max_pool2d(x5, 8), scale_factor=8)
        x5_2 = f.interpolate(f.max_pool2d(x5, 4), scale_factor=4)
        x5_3 = f.interpolate(f.max_pool2d(x5, 2), scale_factor=2)
        x5_4 = f.interpolate(f.max_pool2d(x5, 1), scale_factor=1)
        x6 = torch.cat([x5_1, x5_2, x5_3, x5_4], 1) #128*8*8
        x7 = f.relu(self.conv6(x6)) #64*8*8
        x8 = f.relu(self.conv7(x7)) #32*8*8
        x9 = f.relu(self.conv8(x8)) #8*8*8
        out = self.conv9(x9)
        return out