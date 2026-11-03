"""
Model Module: 2D Convolutional Neural Network (CNN) for Speech Emotion Recognition (SER)
Includes Explainable AI (Grad-CAM Spectrogram Heatmap) and Valence-Arousal Mapping.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard Speech Emotion Classes (compatible with RAVDESS / TESS)
DEFAULT_EMOTIONS = ['Angry', 'Calm', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

EMOTION_EMOJIS = {
    'Angry': '😠',
    'Calm': '😌',
    'Fearful': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprised': '😲'
}

# Russell's Circumplex Model of Affect (Valence: -1.0 to 1.0, Arousal: -1.0 to 1.0)
RUSSELL_COORDINATES = {
    'Angry': (-0.75, 0.85),
    'Calm': (0.60, -0.65),
    'Fearful': (-0.65, 0.80),
    'Happy': (0.80, 0.70),
    'Neutral': (0.00, 0.00),
    'Sad': (-0.80, -0.60),
    'Surprised': (0.35, 0.85)
}


class EmotionCNN(nn.Module):
    """
    4-Stage 2D Deep Convolutional Neural Network designed for Log-Mel Spectrograms.
    Input shape: (Batch, 1, 128, 128)
    """
    def __init__(self, num_classes: int = len(DEFAULT_EMOTIONS)):
        super(EmotionCNN, self).__init__()

        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # (32, 64, 64)
        self.drop1 = nn.Dropout2d(0.20)

        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # (64, 32, 32)
        self.drop2 = nn.Dropout2d(0.25)

        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # (128, 16, 16)
        self.drop3 = nn.Dropout2d(0.30)

        # Block 4: 128 -> 256 (Feature Bottleneck)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1)

        # Dense Classifier Head
        self.fc1 = nn.Linear(256, 128)
        self.drop_fc = nn.Dropout(0.35)
        self.fc_out = nn.Linear(128, num_classes)

        # Hook placeholders for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Block 1
        x = self.drop1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        # Block 2
        x = self.drop2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        # Block 3
        x = self.drop3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        # Block 4
        x = F.elu(self.bn4(self.conv4(x)))

        # Register hook on the final feature map for Grad-CAM Explainable AI
        if x.requires_grad:
            self.activations = x
            h = x.register_hook(self.activations_hook)

        # Global Pooling & Head
        x_pooled = self.global_pool(x)
        x_flat = torch.flatten(x_pooled, 1)
        dense = self.drop_fc(F.elu(self.fc1(x_flat)))
        out = self.fc_out(dense)
        return out


class EmotionPredictor:
    """
    High-level predictor interface that handles model weights loading,
    inference, probability calibration, and Grad-CAM attention heatmap extraction.
    """
    def __init__(self, model_path: str = None, classes: list = None):
        self.classes = classes or DEFAULT_EMOTIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionCNN(num_classes=len(self.classes)).to(self.device)
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                print(f"Loaded emotion model from {model_path}")
            except Exception as e:
                print(f"Failed to load weights from {model_path}: {e}")

        self.model.eval()

    def predict(self, mel_spec_2d: np.ndarray, compute_gradcam: bool = True):
        """
        Runs inference on a normalized 2D Mel-Spectrogram (128, 128).
        
        Returns:
            dict containing:
                - predicted_emotion (str)
                - emoji (str)
                - confidence (float in %)
                - probabilities (dict: emotion -> float in %)
                - valence_arousal (tuple: valence, arousal)
                - attention_map (2D np.ndarray: normalized attention heatmap)
        """
        # Ensure tensor shape (1, 1, 128, 128)
        tensor_in = torch.from_numpy(mel_spec_2d).unsqueeze(0).unsqueeze(0).to(self.device)

        if compute_gradcam:
            tensor_in.requires_grad = True
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            # Backward pass on top class logit for Grad-CAM
            self.model.zero_grad()
            target_logit = logits[0, pred_idx]
            target_logit.backward()

            # Generate Grad-CAM heatmap
            with torch.no_grad():
                if self.model.gradients is not None and self.model.activations is not None:
                    grads = self.model.gradients.detach()
                    activations = self.model.activations.detach()
                    # Global average pool the gradients
                    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                    # Weighted combination of forward activation maps
                    cam = torch.sum(weights * activations, dim=1, keepdim=True)
                    cam = F.relu(cam)
                    # Interpolate to input spectrogram resolution (128, 128)
                    cam = F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False)
                    cam_np = cam.squeeze().cpu().numpy()
                    # Normalize heatmap between 0 and 1
                    cam_min, cam_max = np.min(cam_np), np.max(cam_np)
                    if cam_max - cam_min > 1e-6:
                        attention_map = (cam_np - cam_min) / (cam_max - cam_min)
                    else:
                        attention_map = np.ones((128, 128), dtype=np.float32) * 0.5
                else:
                    attention_map = mel_spec_2d
        else:
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                attention_map = mel_spec_2d

        predicted_emotion = self.classes[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)

        # Dictionary of probabilities
        prob_dict = {
            cls: float(probs[i] * 100.0)
            for i, cls in enumerate(self.classes)
        }

        # Calculate estimated Valence and Arousal using weighted average over coordinates
        v_weighted = sum(probs[i] * RUSSELL_COORDINATES[cls][0] for i, cls in enumerate(self.classes))
        a_weighted = sum(probs[i] * RUSSELL_COORDINATES[cls][1] for i, cls in enumerate(self.classes))

        return {
            "predicted_emotion": predicted_emotion,
            "emoji": EMOTION_EMOJIS.get(predicted_emotion, "🎙️"),
            "confidence": round(confidence, 1),
            "probabilities": {k: round(v, 2) for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)},
            "valence_arousal": (round(float(v_weighted), 3), round(float(a_weighted), 3)),
            "attention_map": attention_map
        }
