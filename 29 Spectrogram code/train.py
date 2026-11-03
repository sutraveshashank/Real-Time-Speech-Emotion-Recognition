"""
Model Training Script
Trains the 2D EmotionCNN on Log-Mel Spectrograms,
performs validation, saves the best weights, and computes academic evaluation metrics
(Confusion Matrix, Accuracy/Loss Curves, Precision, Recall, F1-Score).
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.audio_processor import load_audio
from core.feature_extractor import extract_mel_spectrogram
from core.model import EmotionCNN, DEFAULT_EMOTIONS
from data.download_dataset import create_starter_dataset


def load_dataset_features(dataset_dir: str, target_classes: list = None, sr: int = 22050):
    """
    Scans directory structure (dataset_dir/<emotion>/*.wav),
    loads audio, and extracts 2D Mel-Spectrograms (128, 128).
    """
    classes = target_classes or DEFAULT_EMOTIONS
    class_to_idx = {cls.lower(): i for i, cls in enumerate(classes)}

    X_list = []
    y_list = []
    filepaths = []

    for cls in classes:
        cls_lower = cls.lower()
        search_pattern = os.path.join(dataset_dir, cls_lower, "*.wav")
        files = glob.glob(search_pattern)

        # Also search case-insensitive / uppercase folders
        if not files:
            search_pattern = os.path.join(dataset_dir, cls, "*.wav")
            files = glob.glob(search_pattern)

        for f in files:
            try:
                y, _ = load_audio(f, sample_rate=sr, trim_silence=True)
                mel_spec = extract_mel_spectrogram(y, sr=sr, target_frames=128)
                X_list.append(mel_spec)
                y_list.append(class_to_idx[cls_lower])
                filepaths.append(f)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

    if len(X_list) == 0:
        raise ValueError(f"No valid audio files found in '{dataset_dir}'.")

    X = np.array(X_list, dtype=np.float32)  # Shape: (N, 128, 128)
    y = np.array(y_list, dtype=np.int64)

    # Reshape to (N, 1, 128, 128) for 2D PyTorch Conv
    X = np.expand_dims(X, axis=1)

    print(f"[INFO] Loaded {len(X)} samples with shape {X.shape} across {len(classes)} classes.")
    return X, y, classes


def train_model(
    dataset_dir: str = "dataset_emotions",
    output_dir: str = "models",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 0.001
):
    os.makedirs(output_dir, exist_ok=True)

    # If dataset doesn't exist yet, auto-generate starter dataset
    if not os.path.exists(dataset_dir) or len(glob.glob(os.path.join(dataset_dir, "*", "*.wav"))) == 0:
        print("Dataset not found. Generating starter acoustic emotion dataset...")
        create_starter_dataset(dataset_dir, samples_per_emotion=40)

    # 1. Feature Extraction
    print("\n--- [Step 1/4] Extracting 2D Mel-Spectrogram Features ---")
    X, y, classes = load_dataset_features(dataset_dir)

    # 2. Stratified Train / Val / Test Split
    print("\n--- [Step 2/4] Splitting Train/Val/Test Sets (70/15/15) ---")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)

    # 3. Model Setup
    print("\n--- [Step 3/4] Initializing 2D EmotionCNN & Optimizer ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_weights_path = os.path.join(output_dir, "emotion_cnn.pt")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Starting training on {device} for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%")

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n[SAVED] Best model checkpoint saved to: {best_weights_path}")

    # 4. Final Evaluation on Test Set
    print("\n--- [Step 4/4] Evaluating on Unseen Test Set ---")
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds) * 100.0
    print(f"\n[RESULT] Final Test Accuracy: {test_acc:.2f}%\n")
    print("Classification Report:")
    report_str = classification_report(all_labels, all_preds, target_names=classes)
    print(report_str)

    # Save Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (Test Accuracy: {test_acc:.1f}%)")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save Training Curves Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="darkorange", linestyle="--")
    plt.title("Cross-Entropy Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", color="royalblue")
    plt.plot(history["val_acc"], label="Val Accuracy", color="darkorange", linestyle="--")
    plt.title("Classification Accuracy (%) vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # Save Metadata JSON
    metadata = {
        "classes": classes,
        "sample_rate": 22050,
        "test_accuracy": round(test_acc, 2),
        "epochs_trained": epochs,
        "best_val_loss": round(float(best_val_loss), 4),
        "history": history
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Evaluation artifacts saved in '{output_dir}/'")
    return metadata


if __name__ == '__main__':
    train_model(epochs=20)
