"""
CNN Model — EfficientNet-B0 with Transfer Learning
===================================================
State-of-the-art deep learning model for Blood Group Detection.

Architecture:
    EfficientNet-B0 (ImageNet pretrained)
    └── Custom Classifier Head
        ├── Dropout(0.35)
        ├── Linear(1280 → 512) + BatchNorm1d + ReLU
        ├── Dropout(0.3)
        ├── Linear(512 → 256) + BatchNorm1d + ReLU
        └── Linear(256 → 8)   [8 blood groups]

Why EfficientNet-B0?
    - Only 5.3M parameters (vs VGG16's 138M)
    - ~93% ImageNet top-1 accuracy
    - Scales perfectly with GPU
    - State-of-the-art in medical imaging AI
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import BLOOD_GROUPS, get_model_dir, ensure_dir

NUM_CLASSES = len(BLOOD_GROUPS)
CNN_MODEL_FILENAME = 'cnn_model.pth'
CNN_BEST_WEIGHTS = '_cnn_best_weights.pth'


class BloodGroupCNN(nn.Module):
    """
    EfficientNet-B0 with a custom deep classifier head,
    optimized for fingerprint-based blood group classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(BloodGroupCNN, self).__init__()

        # Load EfficientNet-B0 backbone with ImageNet pretrained weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Feature dimension from EfficientNet-B0's avgpool output
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace default classifier with a deep custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_cnn_model(model: BloodGroupCNN, metrics: dict, class_names: list):
    """
    Save CNN model checkpoint with metadata.

    Args:
        model: Trained BloodGroupCNN instance
        metrics: Dict with accuracy, history, classification_report, etc.
        class_names: List of class names in the order the model was trained on
    """
    model_dir = get_model_dir()
    ensure_dir(model_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'model_type': 'cnn_efficientnet_b0',
        'num_classes': NUM_CLASSES,
        'class_names': class_names,       # ImageFolder sorted class names
        'blood_groups': BLOOD_GROUPS,     # Our standard ordering
    }

    save_path = os.path.join(model_dir, CNN_MODEL_FILENAME)
    torch.save(checkpoint, save_path)
    print(f"\n  [✓] CNN model saved → {save_path}")
    print(f"  [✓] Parameters: {model.count_parameters():,}")


def load_cnn_model(device: torch.device = None):
    """
    Load trained CNN model from disk.

    Args:
        device: torch.device. Auto-detects CUDA if None.

    Returns:
        model: BloodGroupCNN in eval mode
        class_names: List of class names in model's output order
        metrics: Training metrics dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, CNN_MODEL_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at '{model_path}'.\n"
            "Please run: python model/train.py --mode cnn --dataset-path data/sample_fingerprints"
        )

    checkpoint = torch.load(model_path, map_location=device)

    model = BloodGroupCNN(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    class_names = checkpoint.get('class_names', BLOOD_GROUPS)
    metrics = checkpoint.get('metrics', {})

    return model, class_names, metrics


def get_inference_transform():
    """
    Returns the torchvision transform for inference (no augmentation).
    Converts grayscale BMP to 3-channel, resizes to 224x224, and normalizes.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
