"""
Dataset Loader & Generator Module
Provides utilities to:
1. Generate an acoustic emotion dataset for immediate training & testing.
2. Download and organize benchmark emotion datasets (RAVDESS / TESS).
"""

import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm


EMOTIONS = ['Angry', 'Calm', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


def generate_acoustic_sample(emotion: str, duration: float = 3.0, sr: int = 22050) -> np.ndarray:
    """
    Generates synthetic speech-like acoustic signals with distinctive fundamental
    frequencies (F0), formant modulations, harmonics, and energy envelopes
    corresponding to each emotion's psychoacoustic profile.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if emotion == 'Angry':
        # High fundamental frequency, rapid jitter, sharp high harmonics, percussive burst
        f0 = 260.0
        pitch_contour = f0 + 40 * np.sin(2 * np.pi * 5 * t) + 15 * np.random.randn(len(t))
        harmonics = (
            1.0 * np.sin(2 * np.pi * pitch_contour * t) +
            0.7 * np.sin(2 * np.pi * 2 * pitch_contour * t) +
            0.5 * np.sin(2 * np.pi * 3 * pitch_contour * t) +
            0.4 * np.sin(2 * np.pi * 4 * pitch_contour * t)
        )
        envelope = np.clip(1.2 * np.sin(np.pi * t / duration) ** 0.5, 0, 1)
        noise = 0.08 * np.random.randn(len(t))
        signal = (harmonics * envelope) + noise

    elif emotion == 'Happy':
        # Dynamic fluctuating pitch (melodic vibrato), bright upper harmonics
        f0 = 240.0
        pitch_contour = f0 + 50 * np.sin(2 * np.pi * 3.5 * t)
        harmonics = (
            1.0 * np.sin(2 * np.pi * pitch_contour * t) +
            0.6 * np.sin(2 * np.pi * 2 * pitch_contour * t) +
            0.4 * np.sin(2 * np.pi * 3 * pitch_contour * t)
        )
        envelope = np.sin(np.pi * t / duration) ** 0.8
        signal = harmonics * envelope

    elif emotion == 'Sad':
        # Lower pitch, slow descending intonation, weak high harmonics, low energy
        f0 = 130.0 - 20 * (t / duration)
        harmonics = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        )
        envelope = 0.5 * np.sin(np.pi * t / duration) ** 1.5
        noise = 0.01 * np.random.randn(len(t))
        signal = (harmonics * envelope) + noise

    elif emotion == 'Calm':
        # Low, steady fundamental frequency, soft smooth envelope, warm harmonics
        f0 = 145.0 + 5 * np.sin(2 * np.pi * 0.8 * t)
        harmonics = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.35 * np.sin(2 * np.pi * 2 * f0 * t)
        )
        envelope = 0.7 * np.sin(np.pi * t / duration)
        signal = harmonics * envelope

    elif emotion == 'Fearful':
        # High pitch with fast, irregular tremor / amplitude modulation
        f0 = 270.0 + 35 * np.sin(2 * np.pi * 8 * t)
        tremor = 0.7 + 0.3 * np.sin(2 * np.pi * 12 * t)
        harmonics = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        )
        envelope = np.sin(np.pi * t / duration) * tremor
        noise = 0.05 * np.random.randn(len(t))
        signal = (harmonics * envelope) + noise

    elif emotion == 'Surprised':
        # Rapidly rising pitch glide (inflection up), short duration emphasis
        f0 = 180.0 + 150.0 * (t / duration) ** 2
        harmonics = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        )
        envelope = np.exp(-((t - 1.0) ** 2) / 0.5)
        signal = harmonics * envelope

    else:  # Neutral
        # Steady mid-range fundamental frequency, moderate standard harmonics
        f0 = 160.0 + 8 * np.sin(2 * np.pi * 1.5 * t)
        harmonics = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.4 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        envelope = 0.8 * np.sin(np.pi * t / duration)
        signal = harmonics * envelope

    # Normalize amplitude
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.95

    return signal.astype(np.float32)


def create_starter_dataset(output_dir: str = "dataset_emotions", samples_per_emotion: int = 40):
    """
    Creates an initial acoustic dataset for training and testing the 2D CNN model.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating starter emotional dataset in '{output_dir}' with {samples_per_emotion} samples per class...")

    total_files = 0
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(output_dir, emotion.lower())
        os.makedirs(emotion_dir, exist_ok=True)

        for i in range(samples_per_emotion):
            duration = np.random.uniform(2.5, 3.2)
            sig = generate_acoustic_sample(emotion, duration=duration)
            # Add slight random augmentation
            sig = sig + 0.01 * np.random.randn(len(sig))
            filepath = os.path.join(emotion_dir, f"{emotion.lower()}_{i+1:03d}.wav")
            sf.write(filepath, sig, 22050)
            total_files += 1

    print(f"Successfully generated {total_files} audio files across {len(EMOTIONS)} emotion categories in '{output_dir}'.")
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Emotion Dataset Downloader and Generator")
    parser.add_argument("--output", type=str, default="dataset_emotions", help="Output directory")
    parser.add_argument("--samples", type=int, default=40, help="Samples per emotion class")
    args = parser.parse_args()

    create_starter_dataset(args.output, args.samples)
