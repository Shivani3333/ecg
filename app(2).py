import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import resample
import wfdb.processing
import os
import sys

# --- 1. CONFIGURATION (MUST MATCH TRAINING) ---
MODEL_PATH = 'model_B_noisy.keras'
TARGET_FS = 360   # Changed from 125 to 360 to match MIT-BIH training
WINDOW_SIZE = 180 # Changed from 187 to 180 to match model input layer
LABELS = {
    0: "F - Fusion Beat",
    1: "N - Normal Beat",
    2: "S - Supraventricular Ectopic",
    3: "V - Ventricular Ectopic"
}

# --- 2. UTILITY FUNCTIONS ---

@st.cache_resource
def load_ecg_model():
    """Loads the model once to save resources."""
    try:
        # Suppress warnings
        tf.get_logger().setLevel('ERROR')
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model. Make sure '{MODEL_PATH}' is in the same directory.")
        st.error(str(e))
        return None

def preprocess_signal(signal, current_fs):
    """
    Aligns the uploaded signal to the training data format:
    1. Resample to 360Hz.
    2. Z-Score Normalize (matches StandardScaler).
    3. Find Peaks (R-peaks).
    4. Segment into 180-sample windows.
    """
    
    # A. Resample to 360Hz if needed
    if current_fs != TARGET_FS:
        number_of_samples = int(len(signal) * TARGET_FS / current_fs)
        signal = resample(signal, number_of_samples)
        
    # B. Standardization (Z-Score)
    # Your training used StandardScaler, so we use (x - mean) / std
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0: std = 1 
    signal_normalized = (signal - mean) / std
    
    # C. R-Peak Detection (using WFDB as in notebook)
    # We suppress stdout because XQRS prints a lot of logs
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        qrs_detector = wfdb.processing.XQRS(sig=signal, fs=TARGET_FS)
        qrs_detector.detect()
        r_peaks = qrs_detector.qrs_inds
    except Exception:
        r_peaks = []
    finally:
        sys.stdout = old_stdout

    # D. Segmentation
    segments = []
    valid_peaks = []
    
    # 180 window = 90 before, 90 after
    half_window = WINDOW_SIZE // 2
    
    for peak in r_peaks:
        start = peak - half_window
        end = peak + half_window
        
        # Check boundaries to ensure we get exactly 180 points
        if start >= 0 and end <= len(signal_normalized):
            seg = signal_normalized[start:end]
            
            if len(seg) == WINDOW_SIZE:
                segments.append(seg)
                valid_peaks.append(peak)
    
    if len(segments) == 0:
        return None, None, signal_normalized
        
    # Reshape for CNN input: (Batch_Size, 180, 1)
    X = np.array(segments).reshape(-1, WINDOW_SIZE, 1)
    
    return X, valid_peaks, signal_normalized

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="ECG Classifier", layout="wide")
st.title("🫀 ECG Arrhythmia Classifier")
st.markdown(f"**Model Specs:** CNN | Input: {WINDOW_SIZE} samples | Freq: {TARGET_FS}Hz")

# Load Model
model = load_ecg_model()

if model:
    st.success("Model loaded successfully!")
    
    # Sidebar inputs
    st.sidebar.header("Input Settings")
    uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
    input_fs = st.sidebar.number_input("Sampling Rate (Hz) of CSV", value=360, help="MIT-BIH is 360. Check your data source.")

    if uploaded_file is not None:
        try:
            # Read CSV - Assuming single column of values
            df = pd.read_csv(uploaded_file, header=None)
            
            # Flatten to 1D array
            if df.shape[0] > df.shape[1]:
                raw_signal = df.iloc[:, 0].values.astype(float)
            else:
                raw_signal = df.iloc[0, :].values.astype(float)

            # Preview Raw Signal
            st.subheader("Signal Preview")
            st.line_chart(raw_signal[:1000], height=150)
            
            if st.button("Analyze ECG"):
                with st.spinner("Processing signal & predicting..."):
                    
                    # 1. Preprocess
                    X_segments, r_peaks, proc_signal = preprocess_signal(raw_signal, input_fs)
                    
                    if X_segments is not None:
                        # 2. Predict
                        predictions = model.predict(X_segments)
                        pred_classes = np.argmax(predictions, axis=1)
                        pred_labels = [LABELS.get(i, "Unknown") for i in pred_classes]
                        
                        # 3. Display Results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("Detection Timeline")
                            # Create a dataframe for plotting peaks on signal
                            # Note: Plotting the whole signal might be heavy, limit to first few seconds or abnormal beats
                            results_df = pd.DataFrame({
                                "Beat #": range(len(pred_labels)),
                                "Location": r_peaks,
                                "Prediction": pred_labels,
                                "Confidence": [f"{np.max(p)*100:.1f}%" for p in predictions]
                            })
                            st.dataframe(results_df, use_container_width=True)

                        with col2:
                            st.subheader("Summary")
                            counts = pd.Series(pred_labels).value_counts()
                            st.bar_chart(counts)
                            
                            # Alert for abnormalities
                            abnormal_count = len([l for l in pred_labels if not l.startswith('N')])
                            if abnormal_count > 0:
                                st.error(f"⚠️ Detected {abnormal_count} abnormal beats!")
                            else:
                                st.success("✅ All beats appear Normal.")
                            
                    else:
                        st.warning("Signal processed but no valid heartbeats identified. Try checking the Sampling Rate.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
