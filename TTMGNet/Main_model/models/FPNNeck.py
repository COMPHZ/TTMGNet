import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import ASPP
from models.BSDE import BSDE

from models.UCTransNet import *



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 4)  # channel: 16*inplanes ---> 4*inplanes

#         self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
#         self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
#         self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        

#         self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
#         self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
#         self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)
            
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.cca3 = UpBlock_attention(inplanes*8, inplanes*2, nb_Conv=2)
        self.cca2 = UpBlock_attention(inplanes*4, inplanes, nb_Conv=2)
        self.cca1 = UpBlock_attention(inplanes*2, inplanes, nb_Conv=2)

        
        inter_channels = inplanes // 4
        self.out_Conv = nn.Sequential(Conv3Relu(inplanes, inter_channels),
                                  nn.Dropout(0.2),  
                                  nn.Conv2d(inter_channels, 2, (1, 1)))

        
    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats


        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        feature1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        feature2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        feature3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        feature4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 4

        

        encoder_feature = []



    #    encoder_feature.append(feature1)
        xx = self.cca3(feature4, torch.cat([feature3, self.up(feature4)], 1))
        encoder_feature.append(xx)
        xx = self.cca2(xx, torch.cat([feature2, self.up(xx)], 1))
        encoder_feature.append(xx)
        xx = self.cca1(xx, torch.cat([feature1, self.up(xx)], 1))
        encoder_feature.append(xx)
        xx = self.out_Conv(xx)
        encoder_feature.append(xx)
        
        # xx = self.cca3(feature4, torch.cat([feature3, self.up(feature4)], 1))
        # encoder_feature.append(xx)
        # xx = self.cca2(xx, torch.cat([feature2, self.up(xx)], 1))
        # encoder_feature.append(xx)
        # xx = self.cca1(xx, torch.cat([feature1, self.up(xx)], 1))
        # encoder_feature.append(xx)
        # xx = self.out_Conv(xx)
        # encoder_feature.append(xx)
        
        return encoder_feature, xx
