"""
Feature Extractor Module
Extracts 2D Mel-Spectrograms (for Deep Learning 2D CNN),
Linear STFT Spectrograms, FFT Frequency Spectra, MFCCs + Velocity/Acceleration Deltas,
Chroma STFT, Spectral Centroid, Bandwidth, Roll-off, and Zero Crossing Rate.
"""

import numpy as np
import scipy.fftpack
import scipy.signal
import librosa
import pandas as pd


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_frames: int = 128
) -> np.ndarray:
    """
    Computes a standardized 2D Log-Mel Spectrogram suitable for 2D CNN architectures.
    
    Returns:
        np.ndarray: Normalized 2D array of shape (n_mels, target_frames), e.g. (128, 128)
    """
    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    # Convert power spectrogram to decibels (log scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate along time axis to ensure fixed dimensions (n_mels, target_frames)
    current_frames = log_mel.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.min(log_mel))
    elif current_frames > target_frames:
        log_mel = log_mel[:, :target_frames]

    # Normalize to [0, 1] range for stable neural network training & gradient flow
    min_val = np.min(log_mel)
    max_val = np.max(log_mel)
    if max_val - min_val > 1e-6:
        normalized_mel = (log_mel - min_val) / (max_val - min_val)
    else:
        normalized_mel = np.zeros_like(log_mel)

    return normalized_mel.astype(np.float32)


def compute_fft(y: np.ndarray, sr: int):
    """
    Computes the Fast Fourier Transform (FFT) magnitude spectrum up to the Nyquist frequency.
    
    Returns:
        tuple: (freq_axis: np.ndarray, magnitude: np.ndarray)
    """
    N = len(y)
    T = 1.0 / sr
    yf = scipy.fftpack.fft(y)
    freq_axis = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    magnitude = (2.0 / N) * np.abs(yf[:N // 2])
    return freq_axis, magnitude


def compute_linear_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
    """
    Computes Short-Time Fourier Transform (STFT) linear spectrogram in decibels (dB).
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spec_db


def compute_mfcc_suite(y: np.ndarray, sr: int, n_mfcc: int = 13):
    """
    Computes MFCCs along with First-Order Deltas (velocity) and Second-Order Deltas (acceleration).
    Captures transitional acoustics that standard static MFCCs miss.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return {
        "mfcc": mfcc,
        "delta": delta_mfcc,
        "delta2": delta2_mfcc,
        "mfcc_norm": librosa.util.normalize(mfcc, axis=1)
    }


def compute_chroma(y: np.ndarray, sr: int):
    """
    Extracts 12-semitone chroma energy profile (Pitch Classes: C, C#, D, ..., B).
    """
    return librosa.feature.chroma_stft(y=y, sr=sr)


def compute_spectral_features(y: np.ndarray, sr: int):
    """
    Extracts core spectral moments and time-domain dynamics.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "rms": rms
    }


def extract_dsp_summary_table(y: np.ndarray, sr: int) -> pd.DataFrame:
    """
    Produces a comprehensive summary table with Mean, Std Dev, Min, and Max
    across time frames for each signal feature, resolving the single-mean limitation.
    """
    mfcc_data = compute_mfcc_suite(y, sr)
    spectral = compute_spectral_features(y, sr)
    chroma = compute_chroma(y, sr)
    _, fft_mag = compute_fft(y, sr)

    features = {
        "FFT Magnitude": fft_mag,
        "MFCC (Mean Across Coeffs)": np.mean(mfcc_data["mfcc"], axis=0),
        "MFCC Delta (Velocity)": np.mean(mfcc_data["delta"], axis=0),
        "Spectral Centroid (Hz)": spectral["centroid"][0],
        "Spectral Bandwidth (Hz)": spectral["bandwidth"][0],
        "Spectral Roll-off (Hz)": spectral["rolloff"][0],
        "Zero Crossing Rate": spectral["zcr"][0],
        "RMS Energy": spectral["rms"][0],
        "Chroma Mean": np.mean(chroma, axis=0)
    }

    rows = []
    for name, arr in features.items():
        rows.append({
            "Acoustic Feature": name,
            "Mean": float(np.mean(arr)),
            "Std Dev": float(np.std(arr)),
            "Min": float(np.min(arr)),
            "Max": float(np.max(arr))
        })

    return pd.DataFrame(rows)


def compare_sample_rates(y: np.ndarray, orig_sr: int, sample_rates=(8000, 16000, 22050, 44100)) -> pd.DataFrame:
    """
    Replicates and upgrades the multi-sample-rate analysis from the original project notebook.
    Demonstrates the impact of sampling frequency and Nyquist cut-off on spectral metrics.
    """
    records = []
    for sr_target in sample_rates:
        if sr_target == orig_sr:
            y_sub = y
        else:
            y_sub = librosa.resample(y, orig_sr=orig_sr, target_sr=sr_target)

        _, fft_mag = compute_fft(y_sub, sr_target)
        mfcc = librosa.feature.mfcc(y=y_sub, sr=sr_target, n_mfcc=13)
        mfcc_norm = librosa.util.normalize(mfcc, axis=1)
        zcr = librosa.feature.zero_crossing_rate(y_sub)[0]
        chroma = librosa.feature.chroma_stft(y=y_sub, sr=sr_target)
        centroid = librosa.feature.spectral_centroid(y=y_sub, sr=sr_target)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_sub, sr=sr_target)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y_sub, sr=sr_target)[0]

        records.append({
            "Sample Rate (Hz)": sr_target,
            "Nyquist Limit (Hz)": sr_target // 2,
            "FFT Mean Mag": round(float(np.mean(fft_mag)), 5),
            "MFCC Mean": round(float(np.mean(mfcc)), 3),
            "MFCC Norm Mean": round(float(np.mean(mfcc_norm)), 3),
            "ZCR Mean": round(float(np.mean(zcr)), 4),
            "Chroma Mean": round(float(np.mean(chroma)), 4),
            "Centroid (Hz)": round(float(np.mean(centroid)), 1),
            "Bandwidth (Hz)": round(float(np.mean(bandwidth)), 1),
            "Rolloff (Hz)": round(float(np.mean(rolloff)), 1)
        })

    return pd.DataFrame(records)
