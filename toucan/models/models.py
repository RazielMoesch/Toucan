from ..FaceCan import Detection
import torch

def get_detection_model(img_size=(256,256), num_classes=2, pretrained=False):

    model = Detection(img_size, num_classes)
    if pretrained:
        model.load_state_dict(torch.load(""))

    return model