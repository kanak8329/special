"""
CNN Model - Multi-Variant EfficientNet with Transfer Learning
=============================================================
Supports: EfficientNet B0, B3, B4

Architecture:
    EfficientNet-B{N} (ImageNet pretrained)
    +-- Custom Classifier Head
        +-- Dropout(0.40)
        +-- Linear(in_features -> 512) + BatchNorm1d + ReLU
        +-- Dropout(0.35)
        +-- Linear(512 -> 256) + BatchNorm1d + ReLU
        +-- Linear(256 -> 8)   [8 blood groups]
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import BLOOD_GROUPS, get_model_dir, ensure_dir

NUM_CLASSES        = len(BLOOD_GROUPS)
CNN_MODEL_FILENAME = 'cnn_model.pth'
CNN_BEST_WEIGHTS   = '_cnn_best_weights.pth'

# Supported variants: (factory_fn, weights_cls, native_input_size, in_features)
EFFICIENTNET_VARIANTS = {
    'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224, 1280),
    'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 300, 1536),
    'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 380, 1792),
}


class BloodGroupCNN(nn.Module):
    """
    Multi-variant EfficientNet with a custom deep classifier head,
    optimized for fingerprint-based blood group classification.
    Supports: B0 (224px), B3 (300px), B4 (380px).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 model_variant: str = 'efficientnet_b3'):
        super(BloodGroupCNN, self).__init__()

        if model_variant not in EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Unknown variant: {model_variant}. "
                f"Choose from: {list(EFFICIENTNET_VARIANTS.keys())}"
            )

        fn, weights_cls, self.input_size, in_features = EFFICIENTNET_VARIANTS[model_variant]
        self.model_variant = model_variant

        weights = weights_cls if pretrained else None
        self.backbone = fn(weights=weights)

        # Replace default classifier with a deep custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.40),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
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
    """Save CNN model checkpoint with metadata."""
    model_dir = get_model_dir()
    ensure_dir(model_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics'         : metrics,
        'model_type'      : f'cnn_{model.model_variant}',
        'model_variant'   : model.model_variant,
        'num_classes'     : NUM_CLASSES,
        'class_names'     : class_names,
        'blood_groups'    : BLOOD_GROUPS,
    }

    save_path = os.path.join(model_dir, f'cnn_model_{model.model_variant}.pth')
    torch.save(checkpoint, save_path)
    print(f"\n  [OK] CNN model saved -> {save_path}")
    print(f"  [OK] Variant    : {model.model_variant}")
    print(f"  [OK] Parameters : {model.count_parameters():,}")


def load_cnn_model(device: torch.device = None, model_variant: str = 'efficientnet_b3'):
    """Load trained CNN model from disk (by variant)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir  = get_model_dir()
    model_path = os.path.join(model_dir, f'cnn_model_{model_variant}.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at '{model_path}'.\n"
            f"Please run: python model/train.py --mode cnn --model {model_variant} --dataset-path data/sample_fingerprints"
        )

    checkpoint    = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if this is a wrapped checkpoint or raw state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        loaded_variant = checkpoint.get('model_variant', model_variant)
        class_names = checkpoint.get('class_names', BLOOD_GROUPS)
        metrics     = checkpoint.get('metrics', {})
    else:
        state_dict = checkpoint
        loaded_variant = model_variant
        class_names = BLOOD_GROUPS
        metrics = {}

    model = BloodGroupCNN(num_classes=NUM_CLASSES, pretrained=False,
                          model_variant=loaded_variant)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, class_names, metrics


def get_inference_transform(model_variant: str = 'efficientnet_b3'):
    """Returns inference transform for given variant (no augmentation)."""
    from torchvision import transforms

    _, _, input_size, _ = EFFICIENTNET_VARIANTS.get(
        model_variant, EFFICIENTNET_VARIANTS['efficientnet_b0']
    )

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
