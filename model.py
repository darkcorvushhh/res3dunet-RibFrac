# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:29:53 2021

@author: LEGION
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:54:13 2021

@author: LEGION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        return y

class encoder2(nn.Module):
    def __init__(self,out_channel):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class encoder3(nn.Module):
    def __init__(self,out_channel):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias= False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 2,bias = False,dilation = 2)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 4,bias = False,dilation = 4)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class encoder4(nn.Module):
    def __init__(self,out_channel):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 3,bias= False,dilation = 3)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 4,bias = False,dilation = 4)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 5,bias = False,dilation = 5)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder1, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder2, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder3, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn3 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = F.relu(y)
        return y

class decoder4(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(decoder4, self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        return y

class down1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down1, self).__init__()
        self.down = nn.Conv3d(in_channel,out_channel,kernel_size = 2,stride = 2,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.down(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class down2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down2, self).__init__()
        self.down = nn.Conv3d(in_channel,out_channel,kernel_size = 3,stride = 1,padding = 1 ,bias = False)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.down(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channel, out_channel, kernel_size = 2,stride = 2)
        self.bn1 = nn.BatchNorm3d(out_channel)
    def forward(self, x):
        y = self.up(x)
        y = self.bn1(y)
        y = F.relu(y)
        return y

class mapping(nn.Module):
    def __init__(self,n_channel,n_output,scale_factor = (8, 8, 8)):
        super(mapping, self).__init__()
        self.conv = nn.Conv3d(n_channel,n_output,kernel_size = 1,stride = 1)
        self.upsample = nn.Upsample(scale_factor= scale_factor, mode = 'trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.upsample(y)
        return y


#需要提前resize输入图像为256*256*32
class resunet3d(nn.Module):
    def __init__(self , n_channel = 1, n_output = 10,training = True):
        super(resunet3d, self).__init__()
        self.training = training
        self.drop_rate = 0.2
        self.en1 = encoder1(n_channel,16)
        self.en2 = encoder2(32)
        self.en3 = encoder3(64)
        self.en4 = encoder4(128)
        self.de1 = decoder1(128,256)
        self.de2 = decoder2(192,128)
        self.de3 = decoder3(96,64)
        self.de4 = decoder4(48,32)
        self.down1 = down1(16,32)
        self.down2 = down1(32,64)
        self.down3 = down1(64,128)
        self.down4 = down2(128,256)
        self.up2 = up(256,128)
        self.up3 = up(128,64)
        self.up4 = up(64,32)
        self.map1 = mapping(256,n_output,scale_factor=(8, 8 ,8))
        self.map2 = mapping(128,n_output,scale_factor=(4, 4, 4))
        self.map3 = mapping(64,n_output,scale_factor=(2, 2, 2))
        self.map4 = mapping(32,n_output,scale_factor=(1 ,1 ,1))

    def forward(self,x) :
        l1=self.en1(x)+x
        s1=self.down1(l1)
        l2=self.en2(s1)+s1
        l2=F.dropout(l2,self.drop_rate,self.training)
        s2=self.down2(l2)
        l3=self.en3(s2)+s2
        l3=F.dropout(l3,self.drop_rate,self.training)
        s3=self.down3(l3)
        l4=self.en4(s3)+s3
        l4=F.dropout(l4,self.drop_rate,self.training)
        s4=self.down4(l4)
        out=self.de1(l4)+s4
        out=F.dropout(out,self.drop_rate,self.training)
        output1=self.map1(out)
        s6=self.up2(out)
        out=self.de2(torch.cat([s6, l3], dim=1))+s6
        out=F.dropout(out,self.drop_rate,self.training)
        output2=self.map2(out)
        s7=self.up3(out)
        out=self.de3(torch.cat([s7, l2], dim=1))+s7
        out=F.dropout(out,self.drop_rate,self.training)
        output3=self.map3(out)
        s8=self.up4(out)
        out=self.de4(torch.cat([s8, l1], dim=1))+s8
        output4=self.map4(out)
        # print(output1.shape)
        # print(output2.shape)
        # print(output3.shape)
        # print(output4.shape)
        if self.training is True:
            # return output1, output2, output3, output4
            return output4
        else:
            return output4

#与3Dunet相比核心就是在每个encoder阶段构造一个残差块，在decoder阶段加上dropout，同时conv+bn+relu作为一个卷积块，提高收敛速度
#output(B*C*D*H*W)

# model = resunet3d(training = True)
# from torchsummary import summary
# summary(model, input_size=[(1, 64 , 64 , 64)], device="cpu")
# loss用la dice+ce
