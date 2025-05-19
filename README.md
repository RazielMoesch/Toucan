Toucan FaceCan: Object Detection Module Documentation
Overview
The toucan.FaceCan module provides a lightweight and efficient framework for object detection, designed for tasks such as face detection or other bounding box-based detection problems. It leverages a depthwise separable convolutional neural network architecture and anchor-based detection mechanism, implemented using PyTorch. The module includes utilities for anchor generation, IoU computation, dataset handling, and model training.
This documentation covers the installation, usage, and key components of the FaceCan module, including the Detection model, dataset utilities, and training pipeline.
Installation
To use toucan.FaceCan, install the toucan package via pip:
pip install toucan

Ensure you have the following dependencies installed:

torch>=1.9.0
numpy>=1.19.0
Pillow>=8.0.0
albumentations>=1.0.0
tqdm>=4.0.0

You can install these dependencies using:
pip install torch numpy Pillow albumentations tqdm

Getting Started
The toucan.FaceCan module is accessible under the toucan package. To import the module:
import toucan.FaceCan as fc

The primary class for object detection is fc.Detection, which implements the neural network model. Additional utilities are provided for anchor generation, dataset creation, and training.
Creating the Detection Model
To instantiate the Detection model:
model = fc.Detection(img_size=(256, 256), num_classes=2)


Parameters:
img_size: Tuple of (height, width) for input images (default: (256, 256)).
num_classes: Number of classes to detect, including background (default: 2).



The model uses a depthwise separable convolutional backbone (DWBlock) and generates predictions for classification scores and bounding box offsets across multiple feature scales.
Preparing the Dataset
The fc.DetectionDataset class is used to create a dataset for training and evaluation. It supports image transformations using either Albumentations or PyTorch's torchvision.transforms.
Example: Creating a Dataset
from torch.utils.data import DataLoader

# Example data (replace with your actual data)
images = ["path/to/image1.jpg", "path/to/image2.jpg"]
bboxes = [[[50, 50, 100, 100], [150, 150, 200, 200]], [[75, 75, 125, 125]]]  # List of lists of [x_min, y_min, x_max, y_max]
labels = [[1, 1], [1]]  # Corresponding class labels (0 for background, 1 for object)

# Define transformations
transforms = fc.detection_transforms()

# Create dataset
dataset = fc.DetectionDataset(
    images=images,
    bboxes=bboxes,
    labels=labels,
    transform=transforms,
    transform_type="albumentations"
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


Parameters:
images: List of file paths to images.
bboxes: List of lists of bounding boxes in [x_min, y_min, x_max, y_max] format (Pascal VOC).
labels: List of lists of class labels corresponding to the bounding boxes.
transform: Transformation pipeline (e.g., from fc.detection_transforms() or torchvision.transforms).
transform_type: Either "albumentations" or "torch" (default: "albumentations").



The fc.detection_transforms() function provides a default Albumentations pipeline with resizing, augmentation (horizontal flip, brightness/contrast, etc.), and normalization.
Training the Model
The fc.detection_training function handles the training loop, including anchor-based target computation, loss calculation, and validation.
Example: Training the Model
# Initialize model
model = fc.Detection(img_size=(256, 256), num_classes=2)

# Define optimizer (optional, defaults to Adam with lr=3e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
trained_model = fc.detection_training(
    epochs=10,
    model=model,
    train_dl=dataloader,
    val_dl=dataloader,  # Replace with validation DataLoader
    optimizer=optimizer,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tracking_path="training_progress.json"
)


Parameters:
epochs: Number of training epochs.
model: Instance of fc.Detection.
train_dl: Training DataLoader.
val_dl: Validation DataLoader.
optimizer: PyTorch optimizer (default: Adam with specified lr).
lr: Learning rate (default: 3e-3).
device: Device to run training on (default: "cuda" if available, else "cpu").
tracking_path: Path to save training



