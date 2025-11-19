import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import resample
import wfdb
from wfdb import processing
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. CONFIGURATION ---
MODEL_PATH = 'model_B_noisy.keras'
TARGET_FS = 360
WINDOW_SIZE = 180

# Model Classes (Alphabetical Order)
LABELS_MAP = {
    0: "F",
    1: "N",
    2: "S",
    3: "V"
}

# MIT-BIH Annotation Mapping to Model Classes
# Maps standard physionet symbols to our 4 classes
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion
ANNOTATION_MAPPING = {
    'N': 1, 'L': 1, 'R': 1, 'e': 1, 'j': 1,  # Normal variants -> N (1)
    'A': 2, 'a': 2, 'J': 2, 'S': 2,          # SVEB variants -> S (2)
    'V': 3, 'E': 3,                          # VEB variants -> V (3)
    'F': 0,                                  # Fusion -> F (0)
    '/': -1, 'f': -1, 'Q': -1, '?': -1       # Unknown/Paced -> Ignore (-1)
}



def generate_condition_paragraph(y_pred):
    """
    Returns a short, user-friendly paragraph summarizing
    possible heart conditions based on predicted ECG classes.
    """

    if len(y_pred) == 0:
        return "No valid ECG beats were detected, so your heart rhythm could not be analyzed."

    # Count predicted classes
    unique, counts = np.unique(y_pred, return_counts=True)
    count_map = dict(zip(unique, counts))

    n = count_map.get(1, 0)  # Normal
    s = count_map.get(2, 0)  # Supraventricular ectopy
    v = count_map.get(3, 0)  # Ventricular ectopy
    f = count_map.get(0, 0)  # Fusion beats

    total = len(y_pred)

    # --- Build the paragraph ---
    paragraph = "Based on the analysis of your ECG signal, "

    # Mostly normal
    if n / total > 0.90:
        paragraph += (
            "your heart rhythm appears predominantly normal, with regular beats "
            "and no significant signs of arrhythmia."
        )
        return paragraph

    findings = []

    # Supraventricular
    if s > 0:
        findings.append(
            "some supraventricular ectopic beats were detected, which may be associated "
            "with premature atrial contractions or supraventricular arrhythmias"
        )

    # Ventricular
    if v > 0:
        findings.append(
            "ventricular ectopic beats were observed, which can indicate premature ventricular "
            "contractions (PVCs) and may require medical evaluation"
        )

    # Fusion
    if f > 0:
        findings.append(
            "fusion beats were present, meaning the heart's electrical activity came from both "
            "normal and ectopic sources at the same time"
        )

    if findings:
        paragraph += "; ".join(findings) + ". "
        paragraph += (
            "These patterns suggest the presence of irregular cardiac activity, and it may be "
            "helpful to consult a medical professional for further interpretation."
        )
        return paragraph

    # Mild abnormalities
    paragraph += (
        "some irregularities were detected, although they do not clearly correspond to a specific "
        "arrhythmia type. A medical review is recommended."
    )

    return paragraph

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
    """
    Advanced CSV loader with lead-selection logic:
    1. If Lead II + V5 both exist -> sum those
    2. Else if II missing -> sum I + III + V5 (if present)
    3. Else if V1-V5 exist -> sum them
    4. Else -> sum all numeric leads but take abs() amplitude
    """
    try:
        df = pd.read_csv(uploaded_file)
        df = df.select_dtypes(include=[np.number])

        if df.empty:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
            df = df.select_dtypes(include=[np.number])

        if df.empty:
            return None, "No numeric ECG data found in CSV.", False

        # --- Normalize Column Names ---
        cols = [str(c).lower() for c in df.columns]

        def has(name):
            return name.lower() in cols

        # Map common ECG lead naming variations
        lead_aliases = {
            "i": ["i", "lead1", "l1"],
            "ii": ["ii", "lead2", "l2"],
            "iii": ["iii", "lead3", "l3"],
            "v1": ["v1"],
            "v2": ["v2"],
            "v3": ["v3"],
            "v4": ["v4"],
            "v5": ["v5"],
            "v6": ["v6"]
        }

        def get_cols(names):
            return [df.columns[cols.index(alias)]
                    for n in names
                    for alias in lead_aliases.get(n, [])
                    if alias in cols]

        # ------ RULE 1: If II + V5 exist ------
        lead_ii = get_cols(["ii"])
        lead_v5 = get_cols(["v5"])

        if lead_ii and lead_v5:
            msg = "ℹ️ Using **Lead II + V5** (Rule 1)."
            selected = df[lead_ii + lead_v5].sum(axis=1)
            return selected.values.astype(float), msg, True

        # ------ RULE 2: If II missing -> I + III + V5 ------
        lead_i = get_cols(["i"])
        lead_iii = get_cols(["iii"])

        if not lead_ii and (lead_i or lead_iii or lead_v5):
            chosen = lead_i + lead_iii + lead_v5
            chosen = list(dict.fromkeys(chosen))  # unique
            msg = "ℹ️ 'Lead II' not found. Using **I + III + V5** (Rule 2)."
            selected = df[chosen].sum(axis=1)
            return selected.values.astype(float), msg, True

        # ------ RULE 3: Use any of V1–V5 ------
        v_leads = []
        for v in ["v1", "v2", "v3", "v4", "v5"]:
            v_leads += get_cols([v])

        if len(v_leads) >= 2:  # at least two chest leads available
            msg = "ℹ️ Using available **V1–V5** leads (Rule 3)."
            selected = df[v_leads].sum(axis=1)
            return selected.values.astype(float), msg, True

        # ------ RULE 4: Fallback -> sum all numeric leads with abs() ------
        msg = "ℹ️ Using **all numeric leads (abs amplitude)** (Rule 4)."
        selected = df.abs().sum(axis=1)
        return selected.values.astype(float), msg, True

    except Exception as e:
        return None, f"Error reading CSV: {e}", False

def fetch_physionet_record(record_id, db='mitdb'):
    """Fetches signal and annotations from PhysioNet."""
    try:
        # Read record (Signal)
        record = wfdb.rdrecord(record_id, pn_dir=db)
        # Read annotations (Ground Truth)
        annotation = wfdb.rdann(record_id, 'atr', pn_dir=db)
        
        # Handle multiple channels (PhysioNet usually has 2)
        if record.n_sig > 1:
            # Sum all channels as requested
            signal = np.sum(record.p_signal, axis=1)
            msg = f"Fetched Record {record_id}. Merged {record.n_sig} leads (Sum)."
        else:
            signal = record.p_signal.flatten()
            msg = f"Fetched Record {record_id}. Single lead."
            
        return signal, annotation, msg, True
    except Exception as e:
        return None, None, f"Error fetching from PhysioNet: {e}", False

def preprocess_signal(signal, current_fs):
    """
    Preprocessing Pipeline matching Training:
    1. Resample to 360Hz.
    2. Peak Detection (XQRS) on RESAMPLED RAW signal.
    3. Normalize (Z-Score).
    4. Segment from NORMALIZED signal.
    """
    
    # 1. Resample
    if current_fs != TARGET_FS:
        number_of_samples = int(len(signal) * TARGET_FS / current_fs)
        signal = resample(signal, number_of_samples)
    
    # 2. Peak Detection (on RAW signal for better robustness)
    # We perform detection before normalization because XQRS often expects standard ECG amplitudes
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # Suppress wfdb prints
    try:
        qrs_detector = wfdb.processing.XQRS(sig=signal, fs=TARGET_FS)
        qrs_detector.detect()
        r_peaks = qrs_detector.qrs_inds
    except:
        r_peaks = []
    finally:
        sys.stdout = old_stdout

    # 3. Normalize (Z-Score)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0: std = 1
    signal_normalized = (signal - mean) / std
    
    # 4. Segmentation
    segments = []
    valid_peaks = [] # Keep track of which peaks we actually used
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
        return None, [], None
        
    X = np.array(segments).reshape(-1, WINDOW_SIZE, 1)
    return X, valid_peaks, signal_normalized

def evaluate_against_ground_truth(valid_peaks, annotation_obj, predictions_indices):
    """
    Aligns detected peaks with true annotations to calculate accuracy.
    """
    # --- Visualization Functions ---



    true_peaks = annotation_obj.sample
    true_symbols = annotation_obj.symbol
    
    y_true = []
    y_pred = []
    
    # Tolerance window (e.g., 36 samples is 100ms at 360Hz)
    tolerance = int(0.1 * TARGET_FS)
    
    for i, peak in enumerate(valid_peaks):
        # Find closest true peak
        diffs = np.abs(true_peaks - peak)
        min_idx = np.argmin(diffs)
        
        if diffs[min_idx] <= tolerance:
            symbol = true_symbols[min_idx]
            
            # Check if this symbol maps to one of our 4 classes
            if symbol in ANNOTATION_MAPPING:
                class_id = ANNOTATION_MAPPING[symbol]
                
                # -1 means ignore (Unknown/Paced classes not trained on)
                if class_id != -1:
                    y_true.append(class_id)
                    y_pred.append(predictions_indices[i])
                    
    return y_true, y_pred

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="ECG Classifier", layout="wide")
st.title("🫀 Cardiac Anomaly Classifier")

model = load_ecg_model()

if model:
    st.success("Model loaded successfully!")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📂 Upload CSV", "🌐 PhysioNet (MIT-BIH)"])

    # --- TAB 1: CSV UPLOAD ---
    with tab1:
        st.header("Analyze Local File")
        uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])
        input_fs = st.number_input("Sampling Rate (Hz)", value=360, key="csv_fs")

        if uploaded_file is not None:
            signal, msg, success = load_data(uploaded_file)
            
            if success:
                st.markdown(msg) # Show multilead message if applicable
                import matplotlib.pyplot as plt

# Plot first 1000 samples with axis labels
                fig, ax = plt.subplots(figsize=(10, 3))

                t = np.arange(len(signal[:1000])) / input_fs
                ax.plot(t, signal[:1000])

                ax.set_title("ECG Signal (First 1000 samples)")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude (mV)")
                ax.grid(True)

                st.pyplot(fig)

                
                if st.button("Predict CSV"):
                    with st.spinner("Processing..."):
                        X, peaks, _ = preprocess_signal(signal, input_fs)
                        
                        if X is not None:
                            preds = model.predict(X)
                            pred_idxs = np.argmax(preds, axis=1)
                            pred_labels = [LABELS_MAP[i] for i in pred_idxs]
                            
                            # Display Summary
                            st.subheader("Results")
                            counts = pd.Series(pred_labels).value_counts()
                            st.bar_chart(counts)
                            
                            # Highlight Abnormal
                            abnormal = [l for l in pred_labels if l != 'N']
                            if abnormal:
                                st.warning(f"⚠️ Detected {len(abnormal)} Arrhythmic Beats!")
                            else:
                                st.success("✅ Normal Sinus Rhythm Detected")
                                
                            # Detail Table
                            df_res = pd.DataFrame({
                                "Beat Index": range(len(pred_labels)),
                                "Peak Loc": peaks,
                                "Prediction": pred_labels,
                                "Confidence": [f"{np.max(p)*100:.1f}%" for p in preds]
                            })
                            st.dataframe(df_res, use_container_width=True, height=300)
                            condition_text = generate_condition_paragraph(pred_idxs)
                            st.info(condition_text)

                        else:
                            st.error("Could not detect valid heartbeats.")
                            
            else:
                st.error(msg)

    # --- TAB 2: PHYSIONET ---
    # --- TAB 2: PHYSIONET ---
with tab2:
    st.header("Test on MIT-BIH Database")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        record_id = st.text_input("Record ID (e.g., 100, 234)", value="100")
    with col_b:
        st.write("")
        st.write("")
        fetch_btn = st.button("Load Record & Test Accuracy")
        
    if fetch_btn:
        with st.spinner(f"Fetching Record {record_id} from PhysioNet..."):
            sig, ann, msg, ok = fetch_physionet_record(record_id)
            
            if ok:
                st.info(msg)

                fig, ax = plt.subplots(figsize=(10, 3))

                t = np.arange(len(sig[:1000])) / 360  # MIT-BIH 360 Hz
                ax.plot(t, sig[:1000])

                ax.set_title(f"MIT-BIH Record {record_id} – First 1000 Samples")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude (mV)")
                ax.grid(True)

                st.pyplot(fig)

                
                # Predict
                X, peaks, _ = preprocess_signal(sig, 360)  # MIT-BIH is 360Hz
                
                if X is not None:
                    preds = model.predict(X)
                    pred_idxs = np.argmax(preds, axis=1)
                    
                    # Align predictions with ground truth
                    y_true, y_pred_filtered = evaluate_against_ground_truth(peaks, ann, pred_idxs)
                    
                    if y_true and len(y_true) > 0:
                        acc = accuracy_score(y_true, y_pred_filtered)
                        st.metric("Model Accuracy on this Record", f"{acc*100:.2f}%")
                        
                        # ---- FIXED CLASSIFICATION REPORT ----
                        all_labels = [0, 1, 2, 3]
                        all_names = [LABELS_MAP[i] for i in all_labels]

                        report = classification_report(
                            y_true,
                            y_pred_filtered,
                            labels=all_labels,
                            target_names=all_names,
                            zero_division=0
                        )
                        
                        st.text("Classification Report:")
                        st.code(report)
                        condition_text = generate_condition_paragraph(y_pred_filtered)
                        st.info(condition_text)

                        
                    else:
                        st.warning("Could not align detected peaks with ground truth annotations for accuracy.")
                        
                else:
                    st.error("Signal processing failed to extract beats.")
            else:
                st.error(msg)
