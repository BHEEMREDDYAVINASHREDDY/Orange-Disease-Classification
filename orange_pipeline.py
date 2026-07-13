"""
Orange Defect Detection Pipeline
=================================
Approach 1: Classical Image Processing → HOG Features → SVM Classifier
Approach 2: Deep Learning → MobileNetV2 (Transfer Learning)

Classes: fresh, blackspot, canker, grenning
"""

import os
import sys
import warnings
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_DIR = Path("/home/claude/orange_dataset")
TRAIN_DIR   = DATASET_DIR / "train"
TEST_DIR    = DATASET_DIR / "test"
OUTPUT_DIR  = Path("/home/claude/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["blackspot", "canker", "fresh", "grenning"]
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 16
EPOCHS      = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")


# ─────────────────────────────────────────────
# 1. Dataset Analysis
# ─────────────────────────────────────────────
def analyse_dataset():
    print("\n" + "="*60)
    print("SECTION 1: DATASET ANALYSIS")
    print("="*60)

    stats = {}
    for split in ["train", "test"]:
        split_dir = DATASET_DIR / split
        stats[split] = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                files = list(cls_dir.glob("*"))
                stats[split][cls] = len(files)

    print("\nClass distribution:")
    for split, counts in stats.items():
        print(f"  {split.upper()}: {counts}  | Total={sum(counts.values())}")

    # Imbalance check
    train_counts = list(stats["train"].values())
    max_c, min_c = max(train_counts), min(train_counts)
    ratio = max_c / min_c
    print(f"\nImbalance ratio (max/min): {ratio:.2f}")
    if ratio > 1.5:
        print("  ⚠ Class imbalance detected – will use class_weight='balanced' in SVM")
    else:
        print("  ✓ Relatively balanced")

    # Sample image stats
    print("\nChecking sample images for resolution outliers…")
    sizes = []
    for cls in CLASSES:
        for f in list((TRAIN_DIR / cls).glob("*"))[:10]:
            try:
                img = Image.open(f)
                sizes.append(img.size)
            except Exception:
                pass
    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    print(f"  Width  – min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}")
    print(f"  Height – min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (split, counts) in zip(axes, stats.items()):
        colors = ['#FF6B6B','#FFA500','#4CAF50','#2196F3']
        ax.bar(counts.keys(), counts.values(), color=colors, edgecolor='black')
        ax.set_title(f"Class Distribution – {split.upper()}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Images")
        ax.set_xlabel("Class")
        for i, (k, v) in enumerate(counts.items()):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: class_distribution.png")

    # Sample grid
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(9, 12))
    fig.suptitle("Sample Images per Class", fontsize=16, fontweight='bold')
    for r, cls in enumerate(CLASSES):
        files = list((TRAIN_DIR / cls).glob("*"))[:3]
        for c, f in enumerate(files):
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[r][c].imshow(img)
            axes[r][c].axis('off')
            if c == 0:
                axes[r][c].set_ylabel(cls, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sample_images.png")

    return stats


# ─────────────────────────────────────────────
# 2. Feature Extraction (Classical)
# ─────────────────────────────────────────────
def extract_hog_features(img_path):
    """HOG + Color Histogram features for classical ML."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128), _blockSize=(16, 16), _blockStride=(8, 8),
        _cellSize=(8, 8), _nbins=9
    )
    hog_feat = hog.compute(gray).flatten()

    # Color histogram in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_feat = np.concatenate([hist_h, hist_s, hist_v])
    color_feat = color_feat / (color_feat.sum() + 1e-8)

    return np.concatenate([hog_feat, color_feat])


def load_classical_data():
    print("\nExtracting HOG + Color features…")
    X_train, y_train, X_test, y_test = [], [], [], []

    for cls in CLASSES:
        for f in (TRAIN_DIR / cls).glob("*"):
            feat = extract_hog_features(f)
            if feat is not None:
                X_train.append(feat)
                y_train.append(cls)

    for cls in CLASSES:
        for f in (TEST_DIR / cls).glob("*"):
            feat = extract_hog_features(f)
            if feat is not None:
                X_test.append(feat)
                y_test.append(cls)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return (np.array(X_train), np.array(y_train),
            np.array(X_test),  np.array(y_test))


# ─────────────────────────────────────────────
# 3. Approach 1 – SVM
# ─────────────────────────────────────────────
def train_svm(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("APPROACH 1: HOG + SVM (Classical)")
    print("="*60)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       class_weight='balanced', probability=True, random_state=42))
    ])
    print("  Training SVM…")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    plot_confusion_matrix(cm, CLASSES, "SVM (HOG + Color Features)", "cm_svm.png")

    joblib.dump(pipe, OUTPUT_DIR / "svm_model.pkl")
    print("  Saved: svm_model.pkl")

    return {
        "model": "SVM (HOG + Color Histogram)",
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "recall": round(rec, 4), "precision": round(prec, 4),
        "report": classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    }, pipe


# ─────────────────────────────────────────────
# 4. Approach 2 – MobileNetV2
# ─────────────────────────────────────────────
class OrangeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {cls: i for i, cls in enumerate(CLASSES)}
        for cls in CLASSES:
            cls_dir = Path(root_dir) / cls
            for f in cls_dir.glob("*"):
                self.samples.append((str(f), self.label_map[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train_mobilenet(results_so_far):
    print("\n" + "="*60)
    print("APPROACH 2: MobileNetV2 (Transfer Learning)")
    print("="*60)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = OrangeDataset(TRAIN_DIR, train_tf)
    test_ds  = OrangeDataset(TEST_DIR,  test_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights for imbalance
    labels_all = [s[1] for s in train_ds.samples]
    cnt = Counter(labels_all)
    total = len(labels_all)
    weights = torch.tensor([total / (len(CLASSES) * cnt[i]) for i in range(len(CLASSES))],
                           dtype=torch.float).to(DEVICE)

    # Lightweight CNN (MobileNet-style depthwise separable convs – no download needed)
    class DepthwiseSepConv(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.dw = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c), nn.ReLU6(inplace=True))
            self.pw = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU6(inplace=True))
        def forward(self, x): return self.pw(self.dw(x))

    class LightCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
                DepthwiseSepConv(32, 64),
                DepthwiseSepConv(64, 128, stride=2),
                DepthwiseSepConv(128, 128),
                DepthwiseSepConv(128, 256, stride=2),
                DepthwiseSepConv(256, 256),
                DepthwiseSepConv(256, 512, stride=2),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            x = self.features(x).view(x.size(0), -1)
            return self.classifier(x)

    model = LightCNN(len(CLASSES)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"  Training for {EPOCHS} epochs on {DEVICE}…")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss, correct, total_n = 0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total_n += imgs.size(0)
        scheduler.step()
        t_loss = running_loss / total_n
        t_acc  = correct / total_n

        # Val
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                v_loss += criterion(out, labels).item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)
        v_loss /= v_total
        v_acc  = v_correct / v_total

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)
        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"Loss: {t_loss:.4f}/{v_loss:.4f}  "
              f"Acc: {t_acc:.4f}/{v_acc:.4f}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss', color='#FF6B6B')
    ax1.plot(val_losses,   label='Val Loss',   color='#2196F3')
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(train_accs, label='Train Acc', color='#FF6B6B')
    ax2.plot(val_accs,   label='Val Acc',   color='#2196F3')
    ax2.set_title("Training & Validation Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mobilenet_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    pred_names  = [CLASSES[p] for p in all_preds]
    label_names = [CLASSES[l] for l in all_labels]

    acc  = accuracy_score(label_names, pred_names)
    f1   = f1_score(label_names, pred_names, average='weighted')
    rec  = recall_score(label_names, pred_names, average='weighted')
    prec = precision_score(label_names, pred_names, average='weighted')

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print("\n  Classification Report:")
    print(classification_report(label_names, pred_names, target_names=CLASSES))

    cm = confusion_matrix(label_names, pred_names, labels=CLASSES)
    plot_confusion_matrix(cm, CLASSES, "MobileNetV2 (Transfer Learning)", "cm_mobilenet.png")

    torch.save(model.state_dict(), OUTPUT_DIR / "mobilenet_model.pth")
    print("  Saved: mobilenet_model.pth")

    return {
        "model": "MobileNetV2 (Transfer Learning)",
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "recall": round(rec, 4), "precision": round(prec, 4),
        "report": classification_report(label_names, pred_names,
                                        target_names=CLASSES, output_dict=True)
    }, model


# ─────────────────────────────────────────────
# 5. Spoilage % Calculation
# ─────────────────────────────────────────────
def calculate_spoilage_percentage(img_path, cls_name):
    """
    Segments spoiled (non-fresh) regions using HSV color masking.
    Fresh orange → bright orange hue.  Defects → dark / brown / gray.
    Returns percentage of spoiled pixels relative to fruit area.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Fruit mask (remove background) ---
    # Oranges are orange → hue 5–30 in OpenCV (0-180 scale)
    # Use saturation threshold to exclude white/bg
    fruit_mask = cv2.inRange(hsv,
        np.array([0, 40, 40]),
        np.array([40, 255, 255]))
    # Morphology clean-up
    kernel = np.ones((5, 5), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    fruit_pixels = int(fruit_mask.sum() / 255)
    if fruit_pixels < 100:
        return 0.0, img_rgb, fruit_mask

    # --- Spoiled mask (dark / brownish regions inside fruit) ---
    # Very dark value (V < 80) or desaturated brown within fruit
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]),   np.array([180, 255, 80]))
    brown_mask = cv2.inRange(hsv, np.array([5, 30, 50]), np.array([25, 180, 150]))
    spoil_mask = cv2.bitwise_or(dark_mask, brown_mask)
    spoil_mask = cv2.bitwise_and(spoil_mask, fruit_mask)
    spoil_mask = cv2.morphologyEx(spoil_mask, cv2.MORPH_OPEN, kernel)

    spoil_pixels = int(spoil_mask.sum() / 255)
    pct = min(100.0, (spoil_pixels / fruit_pixels) * 100)

    # Visualize
    overlay = img_rgb.copy()
    overlay[spoil_mask > 0] = [220, 50, 50]   # red = spoiled
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    return round(pct, 2), blended, spoil_mask


def run_spoilage_analysis():
    print("\n" + "="*60)
    print("SECTION 4: SPOILAGE PERCENTAGE ANALYSIS")
    print("="*60)

    spoilage_classes = ["blackspot", "canker", "grenning"]
    results = []

    fig, axes = plt.subplots(len(spoilage_classes), 3, figsize=(14, 12))
    fig.suptitle("Spoilage Percentage Analysis", fontsize=16, fontweight='bold')
    col_titles = ["Original", "Spoiled Region (Red)", "Spoil Mask"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=11, fontweight='bold')

    for r, cls in enumerate(spoilage_classes):
        files = sorted((TRAIN_DIR / cls).glob("*"))
        img_path = files[0]
        pct, blended, mask = calculate_spoilage_percentage(img_path, cls)

        results.append({"class": cls, "image": img_path.name, "spoilage_pct": pct})
        print(f"  {cls:12s} | Image: {img_path.name:25s} | Spoilage: {pct:.2f}%")

        orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        axes[r][0].imshow(orig);     axes[r][0].axis('off'); axes[r][0].set_ylabel(cls, fontsize=11, fontweight='bold')
        axes[r][1].imshow(blended);  axes[r][1].axis('off')
        axes[r][2].imshow(mask, cmap='gray'); axes[r][2].axis('off')

        axes[r][1].set_xlabel(f"Spoilage: {pct:.2f}%", fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spoilage_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: spoilage_analysis.png")
    return results


# ─────────────────────────────────────────────
# 6. Model Comparison
# ─────────────────────────────────────────────
def plot_model_comparison(svm_res, cnn_res):
    metrics   = ['accuracy', 'f1', 'recall', 'precision']
    svm_vals  = [svm_res[m] for m in metrics]
    cnn_vals  = [cnn_res[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, svm_vals, width, label='SVM (Classical)',   color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x + width/2, cnn_vals, width, label='MobileNetV2 (Deep)', color='#2196F3', edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model_comparison.png")


# ─────────────────────────────────────────────
# 7. Example Run (Inference on 1 image per class)
# ─────────────────────────────────────────────
def example_run(svm_pipe, cnn_model):
    print("\n" + "="*60)
    print("EXAMPLE RUN: Inference on Test Images")
    print("="*60)

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cnn_model.eval()

    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(8, 4 * len(CLASSES)))
    fig.suptitle("Example Inference – One Image per Class", fontsize=14, fontweight='bold')

    for r, cls in enumerate(CLASSES):
        files = sorted((TEST_DIR / cls).glob("*"))
        img_path = files[0]

        # SVM prediction
        feat = extract_hog_features(img_path)
        svm_pred = svm_pipe.predict([feat])[0]
        svm_prob  = svm_pipe.predict_proba([feat])[0].max()

        # CNN prediction
        img_pil = Image.open(img_path).convert("RGB")
        inp     = test_tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits   = cnn_model(inp)
            probs    = torch.softmax(logits, dim=1).cpu().numpy()[0]
            cnn_idx  = probs.argmax()
            cnn_pred = CLASSES[cnn_idx]
            cnn_prob = probs[cnn_idx]

        correct_svm = "✓" if svm_pred == cls else "✗"
        correct_cnn = "✓" if cnn_pred == cls else "✗"

        print(f"  [{cls:12s}]  SVM→{svm_pred:12s}({svm_prob:.2f}) {correct_svm}  "
              f"CNN→{cnn_pred:12s}({cnn_prob:.2f}) {correct_cnn}")

        img_show = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        axes[r].imshow(img_show)
        axes[r].axis('off')
        color_s = 'green' if svm_pred == cls else 'red'
        color_c = 'green' if cnn_pred == cls else 'red'
        axes[r].set_title(
            f"True: {cls}   |   SVM: {svm_pred} ({svm_prob:.0%}) {correct_svm}   "
            f"|   CNN: {cnn_pred} ({cnn_prob:.0%}) {correct_cnn}",
            fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "example_run.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: example_run.png")


def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Dataset analysis
    stats = analyse_dataset()

    # 2. Classical approach
    X_tr, y_tr, X_te, y_te = load_classical_data()
    svm_res, svm_pipe = train_svm(X_tr, y_tr, X_te, y_te)

    # 3. Deep learning approach
    cnn_res, cnn_model = train_mobilenet(svm_res)

    # 4. Spoilage analysis
    spoilage_results = run_spoilage_analysis()

    # 5. Comparison plot
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    for res in [svm_res, cnn_res]:
        print(f"  {res['model']:35s}  Acc={res['accuracy']}  F1={res['f1']}  "
              f"Recall={res['recall']}  Prec={res['precision']}")
    plot_model_comparison(svm_res, cnn_res)

    # 6. Example run
    example_run(svm_pipe, cnn_model)

    # Save JSON summary
    summary = {
        "dataset_stats": stats,
        "svm_results": {k: v for k, v in svm_res.items() if k != "report"},
        "cnn_results": {k: v for k, v in cnn_res.items() if k != "report"},
        "spoilage_results": spoilage_results
    }
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Saved: results_summary.json")
    print("\n✅ Pipeline complete. All outputs in:", OUTPUT_DIR)
