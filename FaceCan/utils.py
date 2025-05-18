import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_anchors(anchor_sizes: list, aspect_ratios: list):
    anchors = []
    for size in anchor_sizes:
        for ratio in aspect_ratios:
            width = size * np.sqrt(ratio)
            height = size / np.sqrt(ratio)
            anchors.append([width, height])
    return anchors

def generate_anchors(anchors: list, feature_sizes: tuple):
    generated_anchors = []
    feature_map_height, feature_map_width = feature_sizes
    for y in range(feature_map_height):
        for x in range(feature_map_width):
            for anchor in anchors:
                width, height = anchor
                center_x = (x + 0.5) * (1.0 / feature_map_width)
                center_y = (y + 0.5) * (1.0 / feature_map_height)
                generated_anchors.append([center_x, center_y, width, height])
    return np.array(generated_anchors)

def compute_iou(anchor, gt_box):
    x1 = np.maximum(anchor[0] - anchor[2] / 2, gt_box[0] - gt_box[2] / 2)
    y1 = np.maximum(anchor[1] - anchor[3] / 2, gt_box[1] - gt_box[3] / 2)
    x2 = np.minimum(anchor[0] + anchor[2] / 2, gt_box[0] + gt_box[2] / 2)
    y2 = np.minimum(anchor[1] + anchor[3] / 2, gt_box[1] + gt_box[3] / 2)
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    anchor_area = anchor[2] * anchor[3]
    gt_area = gt_box[2] * gt_box[3]
    union = anchor_area + gt_area - intersection
    return intersection / union if union > 0 else 0

def decode_boxes(anchors, offsets):
    boxes = torch.zeros_like(anchors)
    boxes[:, 0] = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]  # center_x = Δx * anchor_w + anchor_x
    boxes[:, 1] = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]  # center_y = Δy * anchor_h + anchor_y
    boxes[:, 2] = torch.exp(offsets[:, 2]) * anchors[:, 2]       # width = exp(Δw) * anchor_w
    boxes[:, 3] = torch.exp(offsets[:, 3]) * anchors[:, 3]       # height = exp(Δh) * anchor_h
    return boxes

def detection_loss(pred_bboxes, true_bboxes, pred_id, true_id, return_separate_losses=False):
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()
    
    cls_loss = cls_loss_fn(pred_id, true_id)
    reg_loss = reg_loss_fn(pred_bboxes, true_bboxes)
    
    if return_separate_losses:
        return cls_loss, reg_loss
    
    return cls_loss + reg_loss