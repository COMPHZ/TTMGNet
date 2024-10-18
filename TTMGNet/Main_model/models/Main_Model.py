import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.TTM import TTM
from models.full_cnn import Full_cnn
from models.block.Base import ChannelChecker
from collections import OrderedDict
from util.common import ScaleInOutput

# from models.decoder_full import FPNNeck
from models.FPNNeck import FPNNeck


# HATNet 继承nn.Module属性。
class MainModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        try:
            print("初始化 inplanes")
            self.inplanes = opt.inplanes

            print("初始化 TTM")
            # self.ttm = TTM(channels=self.inplanes)        
            self.full_cnn = Full_cnn(3, self.inplanes)
            print("初始化 fpnn")
            self.fpnn = FPNNeck(self.inplanes, 2)
            self.pretrain = opt.pretrain
            print("初始化权重")
            self._init_weight()
            print("Main Model 初始化成功")
        except Exception as e:
            print("Main Model 初始化失败:", e)
            raise e

    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        # 首先用hafe 得到四种不同尺寸下的特征图
        
        fa1, fa2, fa3, fa4 = self.full_cnn(xa)
        fb1, fb2, fb3, fb4 = self.full_cnn(xb) 
        
        # fa1, fa2, fa3, fa4 = self.ttm(xa)
        # fb1, fb2, fb3, fb4 = self.ttm(xb)

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  
        encoder_feature, change = self.fpnn(ms_feats)
        out_size=(h_input, w_input)
        # 使用双线性插值法，将输出图片的大小转换成输入图片的大小
        out = F.interpolate(change, size=out_size, mode='bilinear', align_corners=True)
        return encoder_feature, out
   
    def _init_weight(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.pretrain.endswith('.pt'):
            pretrained_dict = torch.load(self.pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                            if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)


# 主要用于组合多个 HATNet 模型的预测结果
class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                print(weight_file[0])
                ckp_path = os.path.join(ckp_path, weight_file[0])
                print(ckp_path)
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            # 检测是否 并行运行
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb):
        xa, xb = self.scale.scale_input((xa, xb))
        out1 = 0
        cd_pred = None

        for i, model in enumerate(self.models_list):
            encoder_feature ,outs = model(xa, xb)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                _, cd_pred = torch.max(out1, 1)

        return encoder_feature, cd_pred

