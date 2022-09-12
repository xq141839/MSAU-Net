import torch.nn as nn
import torch.nn.functional as F
import torch
from model import resnest as msa



class DownAndUp(nn.Module):
    def __init__(self,in_channels, out_channels):
       super(DownAndUp, self).__init__()
       temp = out_channels
       self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, temp, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.Conv2d(temp, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )
    def forward(self, x):
     
        return self.conv1(x)

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels,n_classes):
       super(Model, self).__init__()
       """
       Reference:

        - Zhang, Hang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun et al. "Resnest: Split-attention networks." arXiv preprint arXiv:2004.08955 (2020)
       """
       self.net = msa.resnest18(pretrained=False)
       self.img_channels = img_channels
       self.n_classes = n_classes
       self.maxpool = nn.MaxPool2d(kernel_size=2)
       self.out_conv = nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0)
       self.up_conv1 = Up()
       self.up_conv2 = Up()
       self.up_conv3 = Up()
       self.up_conv4 = Up()
       self.down1 = DownAndUp(img_channels,16)
       self.down2 = self.net.layer1
       self.down3 = self.net.layer2
       self.down4 = self.net.layer3
       self.down5 = self.net.layer4
       self.up1 = self.net.layer6
       self.up2 = self.net.layer8
       self.up3 = self.net.layer10
       self.up4 = self.net.layer12

       
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        
        x3 = self.down2(x2)    
        x4 = self.maxpool(x3)
        
        x5 = self.down3(x4)     
        x6 = self.maxpool(x5)
        
        x7 = self.down4(x6)    
        x8 = self.maxpool(x7)
        
        x9 = self.down5(x8)

        x10 = self.up_conv1(x9,x7)    
        x11 = self.up1(x10)
   
        x12 = self.up_conv2(x11,x5)    
        x13 = self.up2(x12)
      
        x14 = self.up_conv3(x13,x3)   
        x15 = self.up3(x14)
        
        x16 = self.up_conv4(x15,x1)
        x17 = self.up4(x16)
        
        x18 = self.out_conv(x17)
        
        x19 = torch.sigmoid(x18)
        
        return x19