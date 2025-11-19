import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import resample
import wfdb.processing
import os
import sys

# --- 1. CONFIGURATION ---
MODEL_PATH = 'model_B_noisy.keras'
TARGET_FS = 360
WINDOW_SIZE = 180

# Class mapping based on alphabetical order ['F', 'N', 'S', 'V']
LABELS = {
    0: "F - Fusion Beat",
    1: "N - Normal Beat",
    2: "S - Supraventricular Ectopic",
    3: "V - Ventricular Ectopic"
}

# --- 2. UTILITY FUNCTIONS ---

@st.cache_resource
def load_ecg_model():
    try:
        tf.get_logger().setLevel('ERROR')
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model. Make sure '{MODEL_PATH}' is in the same directory.")
        st.error(str(e))
        return None

def load_data(uploaded_file):
    """Robust CSV loader that handles headers automatically."""
    try:
        # 1. Try reading with header inferred (default)
        df = pd.read_csv(uploaded_file)
        
        # Select the first numeric column found
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Found numeric column, use the first one
            return df[numeric_cols[0]].values.astype(float)
        else:
            # No numeric columns found? Maybe header was None but first row was data?
            # Reload with header=None
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
            # Force convert to numeric, coercing errors
            flat_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            return flat_data.values
            
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def preprocess_signal(signal, current_fs):
    # A. Resample
    if current_fs != TARGET_FS:
        number_of_samples = int(len(signal) * TARGET_FS / current_fs)
        signal = resample(signal, number_of_samples)
    
    # B. Normalize (Z-Score)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0: std = 1
    signal_normalized = (signal - mean) / std
    
    # C. R-Peak Detection
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        qrs_detector = wfdb.processing.XQRS(sig=signal_normalized, fs=TARGET_FS)
        qrs_detector.detect()
        r_peaks = qrs_detector.qrs_inds
    except:
        r_peaks = []
    finally:
        sys.stdout = old_stdout

    # D. Segmentation
    segments = []
    valid_peaks = []
    half_window = WINDOW_SIZE // 2
    
    for peak in r_peaks:
        start = peak - half_window
        end = peak + half_window
        
        if start >= 0 and end <= len(signal_normalized):
            seg = signal_normalized[start:end]
            if len(seg) == WINDOW_SIZE:
                segments.append(seg)
                valid_peaks.append(peak)
    
    if len(segments) == 0:
        return None, None, signal_normalized
        
    X = np.array(segments).reshape(-1, WINDOW_SIZE, 1)
    return X, valid_peaks, signal_normalized

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="ECG Classifier", layout="wide")
st.title("🫀 ECG Arrhythmia Classifier")

model = load_ecg_model()

if model:
    st.success("Model loaded successfully!")
    
    st.sidebar.header("Input Settings")
    uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
    input_fs = st.sidebar.number_input("Sampling Rate (Hz)", value=360)

    if uploaded_file is not None:
        # Load data using the robust function
        raw_signal = load_data(uploaded_file)
        
        if raw_signal is not None and len(raw_signal) > 0:
            st.subheader("Signal Preview")
            st.line_chart(raw_signal[:1000], height=150)
            
            if st.button("Analyze ECG"):
                with st.spinner("Processing..."):
                    X_segments, r_peaks, proc_signal = preprocess_signal(raw_signal, input_fs)
                    
                    if X_segments is not None:
                        predictions = model.predict(X_segments)
                        pred_classes = np.argmax(predictions, axis=1)
                        pred_labels = [LABELS.get(i, "Unknown") for i in pred_classes]
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader("Beat-by-Beat Analysis")
                            results_df = pd.DataFrame({
                                "Beat #": range(1, len(pred_labels) + 1),
                                "R-Peak": r_peaks,
                                "Prediction": pred_labels,
                                "Confidence": [f"{np.max(p)*100:.1f}%" for p in predictions]
                            })
                            st.dataframe(results_df, use_container_width=True, height=400)

                        with col2:
                            st.subheader("Summary")
                            counts = pd.Series(pred_labels).value_counts()
                            st.bar_chart(counts)
                            
                            abnormal = [l for l in pred_labels if not l.startswith('N')]
                            if abnormal:
                                st.error(f"⚠️ Found {len(abnormal)} abnormal beats!")
                            else:
                                st.success("✅ Normal Sinus Rhythm")
                    else:
                        st.warning("No heartbeats detected. Check signal quality or sampling rate.")
        else:
            st.error("Could not parse numeric data from CSV. Please ensure it contains ECG signal values.")
