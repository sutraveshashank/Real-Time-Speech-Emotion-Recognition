"""
SoundSense AI: Advanced Spectrogram Analysis & Speech Emotion Recognition Dashboard
Built with Streamlit, PyTorch, Librosa, and Plotly.
"""

import os
import io
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.audio_processor import record_live_audio, load_audio, resample_for_comparison
from core.feature_extractor import (
    extract_mel_spectrogram,
    compute_fft,
    compute_linear_spectrogram,
    compute_mfcc_suite,
    compute_chroma,
    compute_spectral_features,
    extract_dsp_summary_table,
    compare_sample_rates
)
from core.model import EmotionPredictor, DEFAULT_EMOTIONS, EMOTION_EMOJIS, RUSSELL_COORDINATES
from data.download_dataset import generate_acoustic_sample


# Page Configuration
st.set_page_config(
    page_title="SoundSense AI | Spectrogram & Emotion Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        background-color: rgba(30, 41, 59, 0.5);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Cached singleton instance of EmotionPredictor."""
    weights_path = os.path.join("models", "emotion_cnn.pt")
    return EmotionPredictor(model_path=weights_path if os.path.exists(weights_path) else None)


predictor = get_predictor()


# ---------------- SIDEBAR: AUDIO INPUT & CONFIG ----------------
with st.sidebar:
    st.markdown("### 🎙️ Audio Input Source")
    input_mode = st.radio(
        "Select Audio Source:",
        [
            "📁 Upload Audio File",
            "🔴 Record with Microphone",
            "🎵 Existing Project Clip (audio_data1.wav)",
            "🧪 Emotional Presets"
        ],
        index=2 if os.path.exists("audio_data1.wav") else 0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Signal Settings")
    target_sr = st.selectbox("Sample Rate (Hz):", [22050, 16000, 44100, 8000], index=0)
    enable_vad = st.checkbox("Silence Trimming (VAD)", value=True, help="Removes silent dead borders using energy thresholding")

    active_audio = None
    active_sr = target_sr
    audio_label = ""

    # Mode 1: Upload
    if input_mode == "📁 Upload Audio File":
        uploaded_file = st.file_uploader("Upload WAV or MP3 audio:", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            active_audio, active_sr = load_audio(uploaded_file, sample_rate=target_sr, trim_silence=enable_vad)
            audio_label = uploaded_file.name

    # Mode 2: Record Live Mic
    elif input_mode == "🔴 Record with Microphone":
        rec_duration = st.slider("Record Duration (seconds):", 1.0, 5.0, 3.0, 0.5)
        if st.button("🎤 Start Recording Now", type="primary", use_container_width=True):
            with st.spinner(f"Speak now! Recording for {rec_duration} seconds..."):
                try:
                    audio_data, rec_sr, rec_file = record_live_audio(
                        filename="recorded_voice.wav",
                        duration=rec_duration,
                        sample_rate=target_sr
                    )
                    active_audio, active_sr = load_audio(rec_file, sample_rate=target_sr, trim_silence=enable_vad)
                    audio_label = "Live Microphone Recording"
                    st.success("✅ Audio captured successfully!")
                except Exception as e:
                    st.error(f"Recording error: {e}. Check if microphone is accessible.")

    # Mode 3: Existing Project Clip
    elif input_mode == "🎵 Existing Project Clip (audio_data1.wav)":
        default_file = "audio_data1.wav"
        if os.path.exists(default_file):
            active_audio, active_sr = load_audio(default_file, sample_rate=target_sr, trim_silence=enable_vad)
            audio_label = default_file
        else:
            st.warning("`audio_data1.wav` not found in workspace.")

    # Mode 4: Emotional Presets
    else:
        preset_emotion = st.selectbox("Choose Emotion Preset:", DEFAULT_EMOTIONS, index=3)
        if st.button("Generate Acoustic Preset", use_container_width=True):
            sig = generate_acoustic_sample(preset_emotion, duration=3.0, sr=target_sr)
            active_audio, active_sr = load_audio(sig, sample_rate=target_sr, trim_silence=False)
            audio_label = f"Synthesized {preset_emotion} Preset"

    # Playback section
    if active_audio is not None and len(active_audio) > 0:
        st.markdown("---")
        st.markdown("### 🔊 Audio Playback")
        # Write to byte buffer for st.audio
        buf = io.BytesIO()
        sf.write(buf, active_audio, active_sr, format='WAV')
        st.audio(buf.getvalue(), format="audio/wav")
        st.caption(f"Duration: {len(active_audio) / active_sr:.2f}s | Samples: {len(active_audio):,} | SR: {active_sr} Hz")

    st.markdown("---")
    st.markdown("### 🧠 Model Status")
    if predictor.model_loaded:
        st.markdown("<span class='badge-pill' style='color:#4ade80; border-color:#4ade80;'>● Weights: Loaded (CNN)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-pill' style='color:#facc15; border-color:#facc15;'>○ Weights: Initialized (Run train.py)</span>", unsafe_allow_html=True)
    st.caption(f"Compute Device: `{predictor.device}`")


# ---------------- MAIN CONTENT AREA ----------------
st.markdown("<div class='main-header'>🎙️ SoundSense AI: Spectrogram & Emotion Studio</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Deep Learning 2D CNN • Time-Frequency Acoustic Analysis • Multi-Sample Rate Comparison</div>",
    unsafe_allow_html=True
)

if active_audio is None or len(active_audio) == 0:
    st.info("👈 Please select or record an audio sample from the sidebar to begin analysis.")
    st.stop()

# Compute Primary Features
with st.spinner("Analyzing acoustic features..."):
    mel_2d = extract_mel_spectrogram(active_audio, sr=active_sr, target_frames=128)
    prediction = predictor.predict(mel_2d, compute_gradcam=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎭 AI Emotion Recognition",
    "📊 Advanced Spectrogram & DSP Studio",
    "🔬 Multi-Sample Rate Comparison",
    "📘 Model Architecture & College Report"
])


# ---------------- TAB 1: EMOTION RECOGNITION ----------------
with tab1:
    col_hero1, col_hero2, col_hero3 = st.columns([1.2, 1.8, 1.5])

    top_emotion = prediction["predicted_emotion"]
    top_emoji = prediction["emoji"]
    top_conf = prediction["confidence"]
    valence, arousal = prediction["valence_arousal"]

    with col_hero1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 3.5rem;'>{top_emoji}</div>
            <div style='font-size: 1.6rem; font-weight: 700; color: #f8fafc;'>{top_emotion}</div>
            <div style='font-size: 1.1rem; color: #38bdf8; font-weight: 600;'>{top_conf}% Confidence</div>
            <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;'>Predicted Emotion Class</div>
        </div>
        """, unsafe_allow_html=True)

    with col_hero2:
        # Probability Distribution Bar Chart
        probs_df = pd.DataFrame([
            {"Emotion": k, "Probability (%)": v, "Emoji": EMOTION_EMOJIS.get(k, "")}
            for k, v in prediction["probabilities"].items()
        ]).sort_values("Probability (%)", ascending=True)

        fig_probs = px.bar(
            probs_df,
            x="Probability (%)",
            y="Emotion",
            orientation='h',
            text="Probability (%)",
            color="Probability (%)",
            color_continuous_scale="Tealgrn",
            title="Emotion Probability Distribution"
        )
        fig_probs.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False,
            coloraxis_showscale=False
        )
        fig_probs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_probs, use_container_width=True)

    with col_hero3:
        # Russell's Circumplex Model (Valence vs Arousal)
        fig_circumplex = go.Figure()

        # Quadrant background reference lines
        fig_circumplex.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_circumplex.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")

        # Anchor points for all standard emotions
        for emo, (vx, vy) in RUSSELL_COORDINATES.items():
            fig_circumplex.add_trace(go.Scatter(
                x=[vx], y=[vy],
                mode='text+markers',
                text=[f"{EMOTION_EMOJIS[emo]} {emo}"],
                textposition="top center",
                marker=dict(size=8, color="#64748b"),
                hoverinfo="text",
                showlegend=False
            ))

        # Current prediction marker
        fig_circumplex.add_trace(go.Scatter(
            x=[valence], y=[arousal],
            mode='markers+text',
            text=[f"📍 <b>{top_emotion}</b>"],
            textposition="bottom center",
            marker=dict(size=16, color="#f43f5e", symbol="star"),
            name="Current Audio",
            showlegend=False
        ))

        fig_circumplex.update_layout(
            title="Russell's Circumplex Model",
            xaxis=dict(title="Valence (Negative ⇄ Positive)", range=[-1.1, 1.1], zeroline=False),
            yaxis=dict(title="Arousal (Calm ⇄ High Energy)", range=[-1.1, 1.1], zeroline=False),
            height=240,
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig_circumplex, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Explainable AI (XAI): Grad-CAM Spectrogram Attention")
    st.caption("Visualizes which time-frequency acoustic regions the 2D CNN prioritized to make its emotional determination.")

    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        # Normalized Mel Spectrogram
        fig_mel = px.imshow(
            mel_2d,
            origin='lower',
            aspect='auto',
            color_continuous_scale="Inferno",
            labels=dict(x="Time Frames (Normalized)", y="Mel Frequency Filterbanks", color="dB Intensity"),
            title="Input Log-Mel Spectrogram (128x128)"
        )
        fig_mel.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_mel, use_container_width=True)

    with cam_col2:
        # Grad-CAM Heatmap
        fig_cam = px.imshow(
            prediction["attention_map"],
            origin='lower',
            aspect='auto',
            color_continuous_scale="Jet",
            labels=dict(x="Time Frames", y="Mel Filterbanks", color="Gradient Activation"),
            title=f"Grad-CAM Attention Heatmap for Class '{top_emotion}'"
        )
        fig_cam.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_cam, use_container_width=True)


# ---------------- TAB 2: ADVANCED SPECTROGRAM & DSP STUDIO ----------------
with tab2:
    st.markdown("### 📊 Time-Frequency Signal Decomposition")

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        # Waveform Plot
        time_axis = np.linspace(0, len(active_audio) / active_sr, len(active_audio))
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis, y=active_audio,
            mode='lines',
            line=dict(color='#38bdf8', width=1.2),
            name="Amplitude"
        ))
        fig_wave.update_layout(
            title="Time-Domain Waveform & Signal Envelope",
            xaxis_title="Time (seconds)",
            yaxis_title="Normalized Amplitude",
            height=280,
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    with row1_c2:
        # FFT Magnitude Spectrum
        freq_axis, fft_mag = compute_fft(active_audio, active_sr)
        peak_idx = np.argmax(fft_mag)
        peak_freq = freq_axis[peak_idx]

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=freq_axis, y=fft_mag,
            mode='lines',
            line=dict(color='#818cf8', width=1.4),
            name="FFT Spectrum"
        ))
        fig_fft.add_annotation(
            x=peak_freq, y=fft_mag[peak_idx],
            text=f"Dominant: {peak_freq:.1f} Hz",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#f43f5e"
        )
        fig_fft.update_layout(
            title="Fast Fourier Transform (FFT) Magnitude Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis_range=[0, min(8000, active_sr // 2)],  # Zoom in on vocal range
            height=280,
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    st.markdown("---")
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        # MFCC Suite + Velocity Deltas
        mfcc_suite = compute_mfcc_suite(active_audio, active_sr, n_mfcc=13)
        fig_mfcc = px.imshow(
            mfcc_suite["mfcc"],
            origin='lower',
            aspect='auto',
            color_continuous_scale="Plasma",
            labels=dict(x="Frames", y="MFCC Coefficients (1-13)", color="Amplitude"),
            title="Mel-Frequency Cepstral Coefficients (MFCCs)"
        )
        fig_mfcc.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_mfcc, use_container_width=True)

    with row2_c2:
        # Chroma STFT
        chroma = compute_chroma(active_audio, active_sr)
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        fig_chroma = px.imshow(
            chroma,
            origin='lower',
            aspect='auto',
            y=pitches,
            color_continuous_scale="Viridis",
            labels=dict(x="Frames", y="Chroma Pitch Class", color="Energy"),
            title="12-Semitone Chroma STFT (Harmonic Profile)"
        )
        fig_chroma.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_chroma, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Acoustic Statistical Feature Breakdown")
    summary_df = extract_dsp_summary_table(active_audio, active_sr)
    st.dataframe(summary_df.style.format({
        "Mean": "{:.4f}",
        "Std Dev": "{:.4f}",
        "Min": "{:.4f}",
        "Max": "{:.4f}"
    }), use_container_width=True)


# ---------------- TAB 3: MULTI-SAMPLE RATE COMPARISON ----------------
with tab3:
    st.markdown("### 🔬 Multi-Sample Rate Comparison Study")
    st.markdown(
        "Demonstrates the effect of downsampling and the **Nyquist-Shannon Sampling Theorem** on audio bandwidth, "
        "reproducing and upgrading the core experiment from `project.ipynb`."
    )

    test_rates = [8000, 16000, 22050, 44100]
    comp_df = compare_sample_rates(active_audio, active_sr, sample_rates=test_rates)

    st.dataframe(comp_df, use_container_width=True)

    # Plot Comparison Curves
    fig_comp = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Spectral Centroid (Hz) vs SR", "Spectral Rolloff (Hz) vs SR", "FFT Mean Magnitude vs SR")
    )

    fig_comp.add_trace(
        go.Bar(x=[str(r) for r in comp_df["Sample Rate (Hz)"]], y=comp_df["Centroid (Hz)"], marker_color="#38bdf8", name="Centroid"),
        row=1, col=1
    )
    fig_comp.add_trace(
        go.Bar(x=[str(r) for r in comp_df["Sample Rate (Hz)"]], y=comp_df["Rolloff (Hz)"], marker_color="#818cf8", name="Rolloff"),
        row=1, col=2
    )
    fig_comp.add_trace(
        go.Bar(x=[str(r) for r in comp_df["Sample Rate (Hz)"]], y=comp_df["FFT Mean Mag"], marker_color="#c084fc", name="FFT Mag"),
        row=1, col=3
    )

    fig_comp.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    > [!IMPORTANT]
    > **Academic Key Takeaway:**  
    > As the sampling rate decreases from 44,100 Hz to 8,000 Hz:
    > 1. The **Nyquist Limit ($f_s / 2$)** drops from 22,050 Hz down to 4,000 Hz, truncating upper speech formants and fricatives.
    > 2. The **Spectral Roll-off** and **Centroid** drop correspondingly because high-frequency spectral components are removed by the anti-aliasing low-pass filter.
    > 3. Standard speech emotion recognition systems achieve optimal accuracy at **16,000 Hz to 22,050 Hz**, balancing high-frequency resolution with computational efficiency.
    """)


# ---------------- TAB 4: ARCHITECTURE & REPORT ----------------
with tab4:
    st.markdown("### 📘 Model Architecture & College Project Specifications")

    rep_col1, rep_col2 = st.columns([1.2, 1])

    with rep_col1:
        st.markdown("""
        #### 🧠 2D Convolutional Neural Network (EmotionCNN)
        The system treats speech emotion recognition as an **image-like classification problem** on 2D Mel-Spectrograms:
        
        1. **Input Layer:** Log-Mel Spectrogram of shape `(Batch, 1, 128, 128)`
        2. **Conv Block 1:** $32 \\text{ filters } (3\\times 3) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ELU} \\rightarrow \\text{MaxPool}(2\\times 2) \\rightarrow \\text{Dropout}(0.20)$
        3. **Conv Block 2:** $64 \\text{ filters } (3\\times 3) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ELU} \\rightarrow \\text{MaxPool}(2\\times 2) \\rightarrow \\text{Dropout}(0.25)$
        4. **Conv Block 3:** $128 \\text{ filters } (3\\times 3) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ELU} \\rightarrow \\text{MaxPool}(2\\times 2) \\rightarrow \\text{Dropout}(0.30)$
        5. **Conv Block 4:** $256 \\text{ filters } (3\\times 3) \\rightarrow \\text{BatchNorm} \\rightarrow \\text{ELU} \\rightarrow \\text{AdaptiveAvgPool}(1\\times 1)$
        6. **Dense Head:** $\\text{Linear}(256 \\rightarrow 128) \\rightarrow \\text{ELU} \\rightarrow \\text{Dropout}(0.35) \\rightarrow \\text{Linear}(128 \\rightarrow 7)$
        7. **Activation:** Softmax function yielding calibrated probability distribution over emotion classes.
        """)

    with rep_col2:
        st.markdown("""
        #### 📐 Mathematical Formulations for Viva
        - **Short-Time Fourier Transform (STFT):**
          $$X(m, \\omega) = \\sum_{n=-\\infty}^{\\infty} x[n] w[n - m] e^{-j \\omega n}$$
        
        - **Mel-Scale Frequency Conversion:**
          $$m = 2595 \\log_{10} \\left(1 + \\frac{f}{700}\\right)$$
        
        - **Discrete Cosine Transform for MFCCs:**
          $$c_n = \\sum_{k=1}^{K} \\log(S_k) \\cos\\left[\\frac{\\pi n}{K}\\left(k - \\frac{1}{2}\\right)\\right]$$
        """)

    st.markdown("---")
    st.markdown("### 🏆 Evaluation Artifacts")

    cm_path = os.path.join("models", "confusion_matrix.png")
    curves_path = os.path.join("models", "training_curves.png")

    if os.path.exists(cm_path) and os.path.exists(curves_path):
        art_c1, art_c2 = st.columns(2)
        with art_c1:
            st.image(cm_path, caption="Confusion Matrix on Test Dataset", use_container_width=True)
        with art_c2:
            st.image(curves_path, caption="Cross-Entropy Loss & Accuracy vs Training Epochs", use_container_width=True)
    else:
        st.info("💡 Run `python train.py` to train the model and generate high-resolution Confusion Matrix and Training Curve figures.")
