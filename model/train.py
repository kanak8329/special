"""
Model Training Module - CLEAN BASELINE (Step 1)
===============================================
This is the pristine, mathematically stable baseline. 
No experimental multi-heads. No aggressive CutMix.
Just a clean, classic EfficientNet-B3 pipeline.

Usage:
    python model/train.py --mode cnn --dataset-path data/augmented_dataset --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import numpy as np
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import BLOOD_GROUPS, get_model_dir, ensure_dir
from model.cnn_model import BloodGroupCNN, save_cnn_model, CNN_BEST_WEIGHTS

def _generate_cnn_plots(history, all_labels, all_preds, class_names):
    model_dir = get_model_dir()
    ensure_dir(model_dir)
    epochs_ran = len(history['train_loss'])
    x = range(1, epochs_ran + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('EfficientNet-B3 Base Training', color='white', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#444')
            
    axes[0].plot(x, history['train_loss'], color='#e94560', lw=2, label='Train Loss')
    axes[0].plot(x, history['val_loss'], color='#4ecdc4', lw=2, label='Val Loss')
    axes[0].set_title('Loss History', color='white')
    axes[0].legend()
    axes[0].grid(alpha=0.1)

    axes[1].plot(x, history['train_acc'], color='#ffd700', lw=2, label='Train Acc')
    axes[1].plot(x, history['val_acc'], color='#a8ff78', lw=2, label='Val Acc')
    axes[1].set_title('Accuracy History', color='white')
    axes[1].legend()
    axes[1].grid(alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'), facecolor=fig.get_facecolor(), dpi=200)
    plt.close()
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()


def train_cnn(dataset_path: str, epochs: int = 100, batch_size: int = 32, lr: float = 3e-4, model_variant: str = 'efficientnet_b3'):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torch.amp import GradScaler, autocast
    from sklearn.metrics import accuracy_score, classification_report

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 60)
    print("  BLOOD GROUP CNN - CLEAN BASELINE TRAINING")
    print("=" * 60)
    print(f"  Device    : {device}")
    
    # ── Transforms (Stable Baseline) ──
    input_size = 300 if model_variant == 'efficientnet_b3' else 224
    
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Dataset Loading ──
    print(f"  Loading dataset : {dataset_path}")
    full_train_ds = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    full_val_ds   = datasets.ImageFolder(root=dataset_path, transform=val_transform)

    class_names = full_train_ds.classes
    num_classes = len(class_names)
    total = len(full_train_ds)

    np.random.seed(42)
    indices = np.random.permutation(total).tolist()
    val_sz  = max(num_classes, int(0.10 * total))
    test_sz = max(num_classes, int(0.10 * total))
    train_sz = total - val_sz - test_sz

    train_ds = Subset(full_train_ds, indices[:train_sz])
    val_ds   = Subset(full_val_ds,   indices[train_sz:train_sz + val_sz])
    test_ds  = Subset(full_val_ds,   indices[train_sz + val_sz:])

    nw = min(4, mp.cpu_count())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train: {train_sz} | Val: {val_sz} | Test: {test_sz}")

    # ── Model Initialization (Single Head) ──
    model = BloodGroupCNN(num_classes=num_classes, pretrained=True, model_variant=model_variant)
    model = model.to(device)
    print(f"  Model variant: {model_variant}")
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    # Standard Cross Entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None

    # ── Strict Training Loop ──
    best_weights_path = os.path.join(get_model_dir(), CNN_BEST_WEIGHTS)
    ensure_dir(get_model_dir())
    
    # Mixup helpers (soft blend preserving ridges)
    def mixup_data(x, y, alpha=0.2):
        import torch
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam = max(lam, 1 - lam)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def mixup_criterion(crit, pred, ya, yb, lam):
        return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)
    
    best_val_acc = 0.0
    patience = 20
    patience_count = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            apply_mixup = np.random.rand() < 0.50
            if apply_mixup:
                images, ya, yb, lam = mixup_data(images, labels, alpha=0.2)
                if use_amp:
                    with autocast('cuda'):
                        logits = model(images)
                        loss = mixup_criterion(criterion, logits, ya, yb, lam)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images)
                    loss = mixup_criterion(criterion, logits, ya, yb, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                _, pred = logits.max(1)
                train_total += labels.size(0)
                train_correct += (lam * pred.eq(ya).float() + (1-lam) * pred.eq(yb).float()).sum().item()
            else:
                if use_amp:
                    with autocast('cuda'):
                        logits = model(images)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()
                _, pred = logits.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast('cuda'):
                        logits = model(images)
                        loss = criterion(logits, labels)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                _, pred = logits.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        tl = train_loss / len(train_loader)
        vl = val_loss   / len(val_loader)
        ta = 100. * train_correct / train_total
        va = 100. * val_correct   / val_total

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        scheduler.step()

        is_best = va > best_val_acc
        flag = " [* BEST]" if is_best else ""
        print(f"  [Ep {epoch+1:3d}/{epochs}] Loss: {tl:.4f}/{vl:.4f}  Acc: {ta:.1f}%/{va:.1f}%{flag}")

        if is_best:
            best_val_acc = va
            patience_count = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  [!] Early stopping triggered at epoch {epoch+1}")
                break

    # ── Final Test ──
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=False))
    model.eval()
    
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)
    
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if use_amp:
                with autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
            _, pred = logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels_list.extend(labels.numpy())

    from sklearn.metrics import classification_report
    test_acc = accuracy_score(all_labels_list, all_preds) * 100
    print(f"\n  Test Accuracy : {test_acc:.2f}%")
    print(f"  Best Val Acc  : {best_val_acc:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(all_labels_list, all_preds, target_names=class_names, zero_division=0))

    _generate_cnn_plots(history, all_labels_list, all_preds, class_names)

    metrics = {
        'accuracy': test_acc / 100,
        'best_val_accuracy': best_val_acc / 100,
        'history': history,
        'model_type': 'cnn',
        'architecture': 'B3_Baseline',
    }
    save_cnn_model(model, metrics, class_names)
    
    if os.path.exists(best_weights_path):
        os.remove(best_weights_path)
        
    return test_acc / 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cnn')
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default='efficientnet_b3')
    args = parser.parse_args()

    if args.mode.lower() == 'cnn':
        train_cnn(
            dataset_path=args.dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_variant=args.model
        )
    else:
        print("RF mode not supported in this script. Run phase 1 baseline via CNN mode.")
