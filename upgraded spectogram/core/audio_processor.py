"""
Audio Processor Module
Handles live microphone recording, robust audio file loading,
stereo-to-mono conversion, amplitude normalization, and Voice Activity Detection (VAD) silence trimming.
"""

import os
import io
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa


def record_live_audio(filename: str = "temp_recorded.wav", duration: float = 3.0, sample_rate: int = 22050, channels: int = 1):
    """
    Records live audio from the default system microphone.
    
    Args:
        filename: Destination path to save the .wav file.
        duration: Duration in seconds.
        sample_rate: Sampling frequency in Hz (standard 22050 Hz).
        channels: 1 for mono, 2 for stereo.
        
    Returns:
        tuple: (audio_data: np.ndarray, sample_rate: int, filename: str)
    """
    total_samples = int(duration * sample_rate)
    print(f"[REC] Recording started for {duration} seconds at {sample_rate} Hz...")
    
    # Record float32 audio
    audio_data = sd.rec(total_samples, samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Block until recording finishes
    print("[REC] Recording complete!")

    # Flatten if mono
    if channels == 1 and audio_data.ndim > 1:
        audio_data = audio_data.flatten()

    # Save to disk
    sf.write(filename, audio_data, sample_rate)
    return audio_data, sample_rate, filename


def load_audio(file_source, sample_rate: int = 22050, trim_silence: bool = True, top_db: float = 25.0):
    """
    Loads audio from a file path, file-like object, or numpy array.
    Automatically handles stereo-to-mono, resampling, amplitude normalization, and silence trimming.
    
    Args:
        file_source: Path string, bytes/BytesIO buffer, or numpy array.
        sample_rate: Target sample rate in Hz (default 22050 Hz).
        trim_silence: If True, uses VAD energy thresholding to remove silent borders.
        top_db: Decibel threshold below peak for silence trimming.
        
    Returns:
        tuple: (y: np.ndarray, sr: int)
    """
    if isinstance(file_source, np.ndarray):
        y = file_source.astype(np.float32)
        sr = sample_rate
    elif isinstance(file_source, (io.BytesIO, bytes)):
        if isinstance(file_source, bytes):
            file_source = io.BytesIO(file_source)
        y, sr = sf.read(file_source, dtype='float32')
    elif isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"Audio file not found: {file_source}")
        y, sr = librosa.load(file_source, sr=None, mono=True)
    else:
        # Fallback to librosa generic loader
        y, sr = librosa.load(file_source, sr=None, mono=True)

    # Convert to mono if multi-channel
    if y.ndim > 1:
        if y.shape[1] > 1:
            y = np.mean(y, axis=1)
        else:
            y = y.flatten()

    # Resample to target sample rate if needed
    if sample_rate is not None and sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    # Optional: Silence trimming / Voice Activity Detection (VAD)
    if trim_silence and len(y) > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        # Ensure we don't return an empty array if audio was too quiet
        if len(y_trimmed) >= sr * 0.2:  # at least 200ms of audio
            y = y_trimmed

    # Normalize amplitude to [-1.0, 1.0]
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        y = y / max_amp

    return y, sr


def pad_or_truncate(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pads with zeros or truncates an audio array to an exact target length.
    Ensures uniform shape for neural network input.
    """
    current_length = len(y)
    if current_length < target_length:
        padding = target_length - current_length
        return np.pad(y, (0, padding), mode='constant')
    elif current_length > target_length:
        return y[:target_length]
    return y


def resample_for_comparison(y: np.ndarray, orig_sr: int, sample_rates=(8000, 16000, 22050, 44100)):
    """
    Generates versions of the audio at different sample rates for comparative DSP analysis.
    """
    resampled_dict = {}
    for sr in sample_rates:
        if sr == orig_sr:
            resampled_dict[sr] = y
        else:
            resampled_dict[sr] = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return resampled_dict
