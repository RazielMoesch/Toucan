import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .Tracker import Tracker

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
    boxes[:, 0] = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = offsets[:, 1] * anchors[:, 3] + anchors[:, 1] 
    boxes[:, 2] = torch.exp(offsets[:, 2]) * anchors[:, 2]      
    boxes[:, 3] = torch.exp(offsets[:, 3]) * anchors[:, 3]       
    return boxes

def detection_loss(pred_bboxes, true_bboxes, cls_scores, cls_truths):
    
    reg_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()
    
    reg_loss = reg_loss_fn(pred_bboxes, true_bboxes)
    cls_loss = cls_loss_fn(cls_scores, cls_truths)

    
    return reg_loss + cls_loss

class DetectionDataset(Dataset):

    def __init__(self, images, labels, class_scores, transform):

        '''
        Params:
        images: The images for it
        labels: The labels for it
        transform: The transforms for it. MUST contain a T.ToTensor() in the transform.Compose([])
        '''

        self.images = images
        self.labels = labels 
        self.class_scores = class_scores
        self.transform = transform
    

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        score = self.class_scores[idx]

        img = Image.open(img).convert("L")

        img = self.transform(img)

        return img, lbl, score


    def __len__(self):
        return len(self.images)
    
detection_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.1),
            A.CLAHE(p=0.1),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.3,
            label_fields=['labels']
        ))

def detection_training(epochs, model, train_dl, val_dl, optimizer=None, lr=3e-3, device=None, tracking_path="progress.json"):

    if tracking_path:
        tracker = Tracker(tracking_path)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        avg_t_loss = 0
        count = 0
        
        for imgs, lbls, clss in train_dl:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            clss = clss.to(device)
            
            cls_scores, box_offsets = model(imgs)
            anchors = torch.cat(model.anchors).to(device)
            pred_bboxes = decode_boxes(anchors, box_offsets)
            
            loss = detection_loss(pred_bboxes, lbls, cls_scores, clss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_t_loss += loss.item()
            count += 1
        
        model.eval()
        avg_v_loss = 0
        v_count = 0
        
        with torch.no_grad():
            for imgs, lbls, clss in val_dl:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                clss = clss.to(device)
                
                cls_scores, box_offsets = model(imgs)
                pred_bboxes = decode_boxes(anchors, box_offsets)
                val_loss = detection_loss(pred_bboxes, lbls, cls_scores, clss)
                
                avg_v_loss += val_loss.item()
                v_count += 1
        
        train_loss = avg_t_loss / count
        val_loss = avg_v_loss / v_count
        
        if tracking_path:
            metric_names = ["train_loss", "val_loss"]
            metric_values = [train_loss, val_loss]
            tracker.log_metrics(epoch=epoch+1, metric_names=metric_names, metric_values=metric_values)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
        if tracking_path:
            tracker.save_metrics(tracking_path)

    return model


def download_dataset(dataset_name):
    '''
    Current Dataset:

    '''

    



