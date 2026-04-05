"""
Model Training Module — Dual Mode
===================================
Supports two training modes controlled by --mode flag:

  CNN Mode  (--mode cnn): EfficientNet-B0 deep learning, GPU-accelerated,
                           mixed precision, data augmentation, early stopping.
  RF  Mode  (--mode rf):  Random Forest with hand-crafted orientation +
                           wavelet features (legacy, keeps backward compat).

Usage:
    python model/train.py --mode cnn --dataset-path data/sample_fingerprints
    python model/train.py --mode rf  --dataset-path data/sample_fingerprints
"""

import os
import sys
import numpy as np
import cv2
import joblib
import argparse
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.feature_extraction import extract_all_features
from utils.helpers import BLOOD_GROUPS, ensure_dir, get_model_dir, get_sample_dir


# ══════════════════════════════════════════════════════════════════════
#  CNN TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_cnn(dataset_path: str, epochs: int = 50, batch_size: int = 32, lr: float = 3e-4):
    """
    Train EfficientNet-B0 on the fingerprint dataset using GPU acceleration.

    Features:
        - Transfer learning from ImageNet pretrained weights
        - Mixed precision training (FP16) for 2x GPU speed
        - Data augmentation: flip, rotate, jitter, crop
        - CosineAnnealingLR scheduler
        - Early stopping (patience=10)
        - Saves best checkpoint automatically
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torch.amp import GradScaler, autocast

    from model.cnn_model import BloodGroupCNN, save_cnn_model, CNN_BEST_WEIGHTS

    # ── Device setup ──────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "█" * 60)
    print("  BLOOD GROUP CNN — EfficientNet-B0 Training")
    print("█" * 60)
    print(f"\n  🖥️  Device    : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  🎮  GPU       : {gpu_name}")
        print(f"  💾  VRAM      : {vram_gb:.1f} GB")
        # Increase batch size for powerful GPUs
        if vram_gb >= 8:
            batch_size = 64
            print(f"  ⚡  Batch Size: {batch_size} (boosted for VRAM ≥ 8GB)")
    else:
        print("  ⚠️  GPU not found — training on CPU (slower)")

    # ── Transforms ────────────────────────────────────────────────────
    # Training: heavy augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),    # BMP → 3 channels
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])

    # Val/Test: no augmentation, just resize & normalize
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset Loading ───────────────────────────────────────────────
    print(f"\n  Loading dataset from: {dataset_path}")
    full_train_ds = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    full_val_ds   = datasets.ImageFolder(root=dataset_path, transform=val_transform)

    class_names = full_train_ds.classes
    num_classes = len(class_names)
    total = len(full_train_ds)

    print(f"  Classes found ({num_classes}): {class_names}")
    print(f"  Total images  : {total}")

    # ── Train / Val / Test Split (80/10/10) ───────────────────────────
    np.random.seed(42)
    indices = np.random.permutation(total).tolist()
    val_size  = max(num_classes, int(0.10 * total))
    test_size = max(num_classes, int(0.10 * total))
    train_size = total - val_size - test_size

    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    # Use separate ImageFolder instances so each set has its own transform
    train_dataset = Subset(full_train_ds, train_idx)
    val_dataset   = Subset(full_val_ds,   val_idx)
    test_dataset  = Subset(full_val_ds,   test_idx)

    num_workers = min(4, mp.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

    print(f"  Train: {train_size} | Val: {val_size} | Test: {len(test_idx)} samples")
    print(f"  Batch size: {batch_size} | CPU workers: {num_workers}")

    # ── Model ─────────────────────────────────────────────────────────
    model = BloodGroupCNN(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    print(f"\n  EfficientNet-B0 parameters: {model.count_parameters():,}")

    # ── Class Weights (fix imbalance: O-, B- have fewer samples) ─────
    print("\n  Computing class weights to fix imbalance...")
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in full_train_ds:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    for i, (cls, cnt, w) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"    {cls:5s}: {int(cnt):5d} samples → weight {w:.3f}")

    # ── Optimizer: Freeze backbone initially, train only head ─────────
    # Phase 1 (epochs 1-5): freeze ALL backbone, train head only
    for param in model.backbone.features.parameters():
        param.requires_grad = False
    head_params = list(model.backbone.classifier.parameters())

    optimizer = optim.AdamW(head_params, lr=lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.1
    )

    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None

    # ── Training Loop ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  TRAINING LOOP")
    print("═" * 60)

    best_val_acc   = 0.0
    patience_count = 0
    patience       = 15   # increased from 12
    backbone_unfrozen = False
    best_weights_path = os.path.join(get_model_dir(), CNN_BEST_WEIGHTS)
    ensure_dir(get_model_dir())

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # ─ Progressive Unfreezing: unfreeze backbone at epoch 6 ───────
        if epoch == 5 and not backbone_unfrozen:
            print("\n  🔓 Epoch 6: Unfreezing full backbone for fine-tuning...")
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            # Rebuild optimizer with backbone + head at different LRs
            # (avoids the duplicate param group error)
            optimizer = optim.AdamW([
                {'params': model.backbone.features.parameters(), 'lr': lr * 0.05},
                {'params': model.backbone.classifier.parameters(), 'lr': lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
            if use_amp:
                scaler = GradScaler('cuda')
            backbone_unfrozen = True
            print("  🔓 Full model is now trainable.\n")
        # ─ Train ──────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # ─ Validation ─────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss    = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc   = 100. * val_correct   / val_total
        avg_tl    = train_loss / len(train_loader)
        avg_vl    = val_loss   / len(val_loader)

        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step()

        is_best = val_acc > best_val_acc
        flag    = " ⭐ BEST" if is_best else ""
        print(f"  [{epoch+1:3d}/{epochs}] "
              f"Loss: {avg_tl:.4f}/{avg_vl:.4f}  "
              f"Acc: {train_acc:.1f}%/{val_acc:.1f}%"
              f"{flag}")

        if is_best:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # ── Load Best Weights ─────────────────────────────────────────────
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    # ── Test Evaluation ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL TEST EVALUATION")
    print("═" * 60)

    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels_list, all_preds) * 100
    report   = classification_report(all_labels_list, all_preds,
                                     target_names=class_names,
                                     labels=range(len(class_names)),
                                     zero_division=0)
    print(f"\n  Test  Accuracy : {test_acc:.2f}%")
    print(f"  Best  Val Acc  : {best_val_acc:.2f}%\n")
    print("  Classification Report:")
    print("─" * 60)
    print(report)

    # ── Generate Plots ─────────────────────────────────────────────────
    _generate_cnn_plots(history, all_labels_list, all_preds, class_names)

    # ── Save Model ────────────────────────────────────────────────────
    metrics = {
        'accuracy'              : test_acc / 100,
        'best_val_accuracy'     : best_val_acc / 100,
        'classification_report' : report,
        'history'               : history,
        'model_type'            : 'cnn',
    }
    save_cnn_model(model, metrics, class_names)

    # Clean up temp best-weights file
    if os.path.exists(best_weights_path):
        os.remove(best_weights_path)

    return test_acc / 100


def _generate_cnn_plots(history: dict, all_labels: list, all_preds: list, class_names: list):
    """Generate training curves, confusion matrix, and per-class performance bar graphs."""
    model_dir = get_model_dir()
    ensure_dir(model_dir)
    epochs_ran = len(history['train_loss'])
    x = range(1, epochs_ran + 1)

    # 1. Training & Validation Curves ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.plot(x, history['train_loss'], color='#e94560', lw=2, label='Train Loss')
    ax1.plot(x, history['val_loss'],   color='#4ecdc4', lw=2, label='Val Loss', ls='--')
    ax1.set_title('Training & Validation Loss', color='white', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', color='#aaa')
    ax1.set_ylabel('Loss',  color='#aaa')
    ax1.legend(framealpha=0.3, labelcolor='white')
    ax1.grid(alpha=0.15, color='#444')

    ax2.plot(x, history['train_acc'], color='#ffd700', lw=2, label='Train Acc')
    ax2.plot(x, history['val_acc'],   color='#a8ff78', lw=2, label='Val Acc', ls='--')
    ax2.set_title('Training & Validation Accuracy', color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch',       color='#aaa')
    ax2.set_ylabel('Accuracy (%)', color='#aaa')
    ax2.legend(framealpha=0.3, labelcolor='white')
    ax2.grid(alpha=0.15, color='#444')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'),
                dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    # 2. Confusion Matrix ──────────────────────────────────────────────
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix — EfficientNet-B0 CNN', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision / Recall / F1 Bars ──────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )
    x_pos = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_pos - w, precision, w, label='Precision', color='#4fc3f7', alpha=0.9)
    ax.bar(x_pos,     recall,    w, label='Recall',    color='#a5d6a7', alpha=0.9)
    ax.bar(x_pos + w, f1,        w, label='F1 Score',  color='#ef9a9a', alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names)
    ax.set_title('CNN Performance Metrics per Blood Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (0–1)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Error Rate per class ──────────────────────────────────────────
    error_rate = 1.0 - f1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(class_names, error_rate, marker='o', color='tomato', lw=2)
    ax.fill_between(class_names, error_rate, alpha=0.15, color='tomato')
    ax.set_title('Error Rate per Blood Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Error Rate (1 − F1)')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'error_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("  [✓] Training curves, Confusion Matrix, Precision/Recall & Error Rate graphs saved.")


# ══════════════════════════════════════════════════════════════════════
#  RANDOM FOREST TRAINING (Legacy — kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════

def generate_synthetic_fingerprint(args):
    """Generate synthetic fingerprint image for RF fallback."""
    blood_group_idx, sample, img_size = args
    seed = blood_group_idx * 10000 + sample
    np.random.seed(seed)

    img = np.zeros((img_size, img_size), dtype=np.float64)
    base_freq  = 0.04 + (blood_group_idx * 0.008)
    base_angle = (blood_group_idx * np.pi / 8) + np.random.uniform(-0.1, 0.1)
    noise_level  = 0.05 + (blood_group_idx % 4) * 0.02
    curve_factor = 0.001 + (blood_group_idx % 5) * 0.0005

    y, x = np.mgrid[0:img_size, 0:img_size]
    xc = x - img_size / 2
    yc = y - img_size / 2

    freq  = base_freq  + np.random.uniform(-0.005, 0.005)
    angle = base_angle + np.random.uniform(-0.15,  0.15)
    x_rot = xc * np.cos(angle) + yc * np.sin(angle)
    y_rot = -xc * np.sin(angle) + yc * np.cos(angle)
    curv  = curve_factor * (x_rot ** 2 + y_rot ** 2)
    ridges = np.sin(2 * np.pi * freq * x_rot + curv)

    sa = angle + np.pi / 4 + np.random.uniform(-0.2, 0.2)
    xr2 = xc * np.cos(sa) + yc * np.sin(sa)
    ridges += 0.3 * np.sin(2 * np.pi * freq * 1.5 * xr2)

    mask = np.exp(-((xc / (img_size * 0.38)) ** 2 + (yc / (img_size * 0.45)) ** 2) * 2)
    img  = (ridges * mask) + np.random.randn(img_size, img_size) * noise_level
    num_min = 15 + blood_group_idx * 3
    for _ in range(num_min):
        mx = np.random.randint(img_size // 4, 3 * img_size // 4)
        my = np.random.randint(img_size // 4, 3 * img_size // 4)
        cv2.circle(img, (mx, my), np.random.randint(1, 3), np.random.uniform(0.5, 1.0), -1)

    img = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return img, blood_group_idx


def process_image(img, label):
    try:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        features, _ = extract_all_features(enhanced)
        return features, label
    except Exception:
        return None, label


def process_image_path(args):
    filepath, label = args
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, label
    return process_image(img, label)


def load_real_dataset_rf(dataset_path: str):
    """Load dataset for RF training (grayscale feature extraction)."""
    print("=" * 60)
    print(f"  Loading RF Dataset from: {dataset_path}")
    print("=" * 60)

    # Walk to find actual class folders
    for root, dirs, files in os.walk(dataset_path):
        normalized = [d.replace('_pos', '+').replace('_neg', '-') for d in dirs]
        if any(c in BLOOD_GROUPS for c in normalized):
            dataset_path = root
            break

    tasks = []
    for idx, bg in enumerate(BLOOD_GROUPS):
        possible_names = [bg, bg.replace('+', '_pos').replace('-', '_neg')]
        for name in possible_names:
            p = os.path.join(dataset_path, name)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        tasks.append((os.path.join(p, f), idx))
                break

    if not tasks:
        raise ValueError("No valid images found in the dataset directory.")

    print(f"Found {len(tasks)} images. Extracting features using multiprocessing...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_image_path, tasks)

    X = [feat for feat, _ in results if feat is not None]
    y = [lbl  for feat, lbl in results if feat is not None]
    print(f"  Successfully processed {len(X)} samples.")
    return np.array(X), np.array(y)


def generate_dataset_rf(samples_per_class: int = 200, img_size: int = 256, save_samples: bool = True):
    """Generate synthetic dataset for RF training."""
    print("=" * 60)
    print(f"  Generating Synthetic Dataset ({samples_per_class} per class)")
    print("=" * 60)
    if save_samples:
        ensure_dir(get_sample_dir())

    tasks = [(idx, s, img_size) for idx in range(len(BLOOD_GROUPS)) for s in range(samples_per_class)]
    with mp.Pool(mp.cpu_count()) as pool:
        gen_results = pool.map(generate_synthetic_fingerprint, tasks)
        feat_results = pool.starmap(process_image, gen_results)

    X = [f for f, _ in feat_results if f is not None]
    y = [l for f, l in feat_results if f is not None]
    print(f"  Dataset generated: {len(X)} samples")
    return np.array(X), np.array(y)


def train_rf(X: np.ndarray, y: np.ndarray):
    """Train Random Forest model."""
    print("\n" + "=" * 60)
    print("  Training Random Forest Model")
    print("=" * 60)

    try:
        from imblearn.over_sampling import SMOTE
        min_samples = int(np.min(np.bincount(y)))
        if min_samples > 1:
            k = min(5, min_samples - 1)
            X, y = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
            print(f"  SMOTE augmented → {X.shape[0]} samples")
    except ImportError:
        print("  Warning: imbalanced-learn not installed. Skipping SMOTE.")

    if X.shape[0] < len(BLOOD_GROUPS) * 5:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=300, max_depth=25,
                                   min_samples_split=5, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n  RF Accuracy: {acc * 100:.2f}%")

    report = classification_report(y_te, y_pred,
                                   labels=range(len(BLOOD_GROUPS)),
                                   target_names=BLOOD_GROUPS, zero_division=0)
    print(report)

    # Save RF plots
    model_dir = get_model_dir()
    ensure_dir(model_dir)
    conf_mat = confusion_matrix(y_te, y_pred, labels=range(len(BLOOD_GROUPS)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=BLOOD_GROUPS, yticklabels=BLOOD_GROUPS, ax=ax)
    ax.set_title('Confusion Matrix — Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save RF model
    joblib.dump(model,  os.path.join(model_dir, 'blood_group_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump({'accuracy': acc, 'classification_report': report, 'model_type': 'rf'},
                os.path.join(model_dir, 'metrics.pkl'))
    print(f"\n  RF model saved → {model_dir}")
    return model, scaler, {'accuracy': acc, 'classification_report': report}


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Blood Group Detection — Model Training")
    parser.add_argument('--mode', type=str, default='cnn',
                        choices=['cnn', 'rf'],
                        help="Training mode: 'cnn' (EfficientNet-B0) or 'rf' (Random Forest)")
    parser.add_argument('--dataset-path', type=str, default=None,
                        help="Path to fingerprint dataset directory")
    parser.add_argument('--epochs',     type=int,   default=50,
                        help="CNN training epochs (default: 50)")
    parser.add_argument('--batch-size', type=int,   default=32,
                        help="CNN batch size (default: 32, auto-adjusted for GPU)")
    parser.add_argument('--lr',         type=float, default=3e-4,
                        help="CNN learning rate (default: 3e-4)")
    parser.add_argument('--samples',    type=int,   default=300,
                        help="RF: synthetic samples per class")
    args = parser.parse_args()

    ensure_dir(get_model_dir())

    # ── CNN Mode ──────────────────────────────────────────────────────
    if args.mode == 'cnn':
        dataset_path = args.dataset_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'sample_fingerprints'
        )
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path not found: {dataset_path}")
            sys.exit(1)

        final_acc = train_cnn(
            dataset_path=dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        print("\n" + "█" * 60)
        print("  CNN TRAINING COMPLETE!")
        print(f"  Final Test Accuracy: {final_acc * 100:.2f}%")
        print("  Model: model/saved_model/cnn_model.pth")
        print("█" * 60 + "\n")

    # ── RF Mode ──────────────────────────────────────────────────────
    else:
        if args.dataset_path and os.path.exists(args.dataset_path):
            X, y = load_real_dataset_rf(args.dataset_path)
        else:
            if args.dataset_path:
                print(f"Warning: Dataset path not found. Falling back to synthetic.")
            X, y = generate_dataset_rf(samples_per_class=args.samples)

        if len(X) == 0:
            print("Error: No data to train on.")
            sys.exit(1)

        train_rf(X, y)

        print("\n" + "█" * 60)
        print("  RF TRAINING COMPLETE!")
        print("█" * 60 + "\n")


if __name__ == '__main__':
    mp.freeze_support()   # Required for Windows multiprocessing
    main()
