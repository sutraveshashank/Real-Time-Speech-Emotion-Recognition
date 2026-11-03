# 🎙️ SoundSense AI: Advanced Spectrogram Analysis & Speech Emotion Recognition

An advanced audio signal processing and deep learning project that elevates acoustic feature extraction into real-time **Speech Emotion Recognition (SER)** using a **2D Convolutional Neural Network (CNN)** on Log-Mel Spectrograms, complete with **Explainable AI (Grad-CAM Attention)** and an interactive **Streamlit Web Studio**.

---

## 🌟 Key Highlights & Upgrades Over the Base Code

| Component | Original Project (`project.ipynb`) | SoundSense AI Upgraded System |
| :--- | :--- | :--- |
| **Representation** | Collapsed 1D scalar means (`np.mean`) | **2D Log-Mel Spectrograms (128x128)** retaining full time-frequency dynamics |
| **Model** | Incomplete Random Forest on flattened vectors | **4-Stage 2D Deep CNN (`EmotionCNN`)** with BatchNorm, Dropout & Adaptive Pooling |
| **Explainable AI (XAI)** | None | **Grad-CAM Spectrogram Heatmap** highlighting frequencies driving predictions |
| **Acoustic Features** | Basic FFT, static MFCC, ZCR | **FFT Spectrum, Mel-Spectrogram, MFCC + Deltas + Delta-Deltas, Chroma STFT, Spectral Centroid, Bandwidth, Roll-off, RMS Energy** |
| **User Interface** | Static Jupyter notebook plots | **Interactive Streamlit Web Dashboard (`app.py`)** with live microphone recording & audio upload |
| **Multi-Sample Rate Study** | Basic DataFrame | Upgraded comparative study with interactive bar charts & Nyquist cutoff analysis |

---

## 📂 Project Architecture

```text
29 Spectrogram code/
│
├── core/
│   ├── __init__.py
│   ├── audio_processor.py       # Live mic recording, VAD silence trimming, audio loaders
│   ├── feature_extractor.py     # 2D Mel-Spectrogram, FFT, MFCC+Deltas, Chroma, Spectral metrics
│   └── model.py                 # 2D EmotionCNN, Grad-CAM attention, Russell's Circumplex mapping
│
├── data/
│   └── download_dataset.py      # Dataset loader & synthetic emotional acoustic generator
│
├── models/                      # Saved PyTorch checkpoints & evaluation artifacts
│   ├── emotion_cnn.pt           # Trained CNN weights
│   ├── metadata.json            # Model configuration & training metrics
│   ├── confusion_matrix.png     # Test set confusion matrix
│   └── training_curves.png      # Loss & accuracy curves
│
├── app.py                       # Flagship Streamlit interactive web application
├── train.py                     # Automated training pipeline with validation & metrics
├── project_upgraded.ipynb       # Clean, production-ready Jupyter Notebook
└── requirements.txt             # Project dependencies
```

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web Dashboard
Run the Streamlit application from your terminal:
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.
- Choose **"🔴 Record with Microphone"** to test your voice live, or **"📁 Upload Audio File"** to test any WAV/MP3 file.
- Inspect the **AI Emotion Prediction**, **Grad-CAM Spectrogram Attention**, **Acoustic Studio**, and **Multi-Sample Rate Comparison**.

### 2. Retrain the Deep Learning Model
To retrain the 2D CNN model from scratch:
```bash
python train.py
```
This script will:
1. Load or generate acoustic dataset files.
2. Extract standardized 2D Mel-Spectrograms ($128 \times 128$).
3. Train `EmotionCNN` with early stopping and AdamW optimization.
4. Output the **Test Accuracy**, **Classification Report**, and save `confusion_matrix.png` and `training_curves.png`.

---

## 📐 Academic & Theory Reference (For College Viva & Project Report)

### 1. Short-Time Fourier Transform (STFT)
$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$
Where $w[n]$ is a Hanning/Hamming analysis window shifting across discrete time $m$.

### 2. Mel-Scale Frequency Conversion
Human hearing perceives pitch logarithmically rather than linearly. The Mel scale models the non-linear response of human cochlea:
$$m = 2595 \log_{10} \left(1 + \frac{f}{700}\right)$$

### 3. Delta and Delta-Delta MFCCs
Static MFCCs capture spectral envelope texture, while first and second derivatives capture temporal velocity and acceleration of vocal tract articulators:
$$d_t = \frac{\sum_{n=1}^{N} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}$$

### 4. Russell's Circumplex Model of Affect
Emotions are mapped onto a 2D continuous coordinate plane:
- **Valence (X-axis):** Negative (unpleasant) $\longleftrightarrow$ Positive (pleasant)
- **Arousal (Y-axis):** Deactivated (calm/sleepy) $\longleftrightarrow$ Activated (high energy/excitement)
