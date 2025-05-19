import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_anchors, generate_anchors

class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DWBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Detection(nn.Module):
    def __init__(self, img_size=(256, 256), num_classes=2):
        super(Detection, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.anchor_sizes = [[16, 32, 64], [64, 128, 256]]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.feature_sizes = [(16, 16), (8, 8)]
        
        self.anchors = []
        self.num_anchors_per_scale = []
        for sizes, feat_size in zip(self.anchor_sizes, self.feature_sizes):
            base_anchors = create_anchors(sizes, self.aspect_ratios)
            anchors = generate_anchors(base_anchors, feat_size, image_size=self.img_size[0])
            self.anchors.append(torch.tensor(anchors, dtype=torch.float32))
            self.num_anchors_per_scale.append(len(base_anchors))
        
        self.backbone = nn.Sequential(
            DWBlock(1, 32, stride=2),
            DWBlock(32, 64, stride=1),
            DWBlock(64, 128, stride=2),
            DWBlock(128, 256, stride=2),
            DWBlock(256, 512, stride=2),
            DWBlock(512, 1024, stride=2)
        )
        
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors_per_scale[0] * num_classes, kernel_size=1),
            nn.Conv2d(1024, self.num_anchors_per_scale[1] * num_classes, kernel_size=1)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(512, self.num_anchors_per_scale[0] * 4, kernel_size=1),
            nn.Conv2d(1024, self.num_anchors_per_scale[1] * 4, kernel_size=1)
        ])
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:
                features.append(x)
            elif i == 5:
                features.append(x)
        
        cls_scores = []
        box_offsets = []
        batch_size = x.size(0)
        for feat, cls_head, reg_head, num_anchors in zip(
            features, self.cls_heads, self.reg_heads, self.num_anchors_per_scale
        ):
            cls = cls_head(feat).permute(0, 2, 3, 1).contiguous()
            h, w = feat.shape[2], feat.shape[3]
            cls = cls.view(batch_size, h * w * num_anchors, self.num_classes)
            cls_scores.append(cls)
            
            reg = reg_head(feat).permute(0, 2, 3, 1).contiguous()
            reg = reg.view(batch_size, h * w * num_anchors, 4)
            box_offsets.append(reg)
        
        cls_scores = torch.cat(cls_scores, dim=1)
        box_offsets = torch.cat(box_offsets, dim=1)
        return cls_scores, box_offsets