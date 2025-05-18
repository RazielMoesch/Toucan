import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_anchors, generate_anchors, compute_iou




class StartingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(StartingBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(self.convs(h) + x)

class MiddleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(MiddleBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)
        return self.act(self.convs(h))

class DetectionCan(nn.Module):
    def __init__(self, img_size=(256, 256), num_classes=2):
        super(DetectionCan, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.anchor_sizes = [
            [8, 16, 32],     
            [32, 64, 128],   
            [64, 128, 256]   
        ]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.feature_sizes = [(16, 16), (8, 8), (4, 4)]
        
        self.anchors = []
        self.num_anchors_per_scale = []
        for sizes, feat_size in zip(self.anchor_sizes, self.feature_sizes):
            base_anchors = create_anchors(sizes, self.aspect_ratios)
            anchors = generate_anchors(base_anchors, feat_size)
            self.anchors.append(torch.tensor(anchors, dtype=torch.float32))
            self.num_anchors_per_scale.append(len(base_anchors))
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),
            )
        
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(1024 , n * num_classes, kernel_size=1) for n in self.num_anchors_per_scale
        ])
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(1024 , n * 4, kernel_size=1) for n in self.num_anchors_per_scale
        ])
    
    def forward(self, x):
        features = []
        curr = x
        for layer in self.backbone:
            curr = layer(curr)
            if curr.shape[2] in [16, 8, 4]:
                features.append(curr)
        
        cls_scores = []
        box_offsets = []
        batch_size = x.size(0)
        for i, (feat, cls_head, reg_head, num_anchors) in enumerate(zip(
            features, self.cls_heads, self.reg_heads, self.num_anchors_per_scale)):
            cls = cls_head(feat).permute(0, 2, 3, 1).contiguous()  
            h, w = feat.shape[2], feat.shape[3]
            cls = cls.view(batch_size, h, w, num_anchors, self.num_classes)
            cls = cls.view(batch_size, h * w * num_anchors, self.num_classes)
            cls_scores.append(cls)
            
            
            reg = reg_head(feat).permute(0, 2, 3, 1).contiguous()  
            reg = reg.view(batch_size, h, w, num_anchors, 4)
            reg = reg.view(batch_size, h * w * num_anchors, 4)
            box_offsets.append(reg)
        
        cls_scores = torch.cat(cls_scores, dim=1) 
        box_offsets = torch.cat(box_offsets, dim=1)  
        return cls_scores, box_offsets
    

    