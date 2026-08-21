"""
================================================================================
Scam Detection Classifier Training Pipeline (XGBoost + TF-IDF)
================================================================================

This script trains a high-accuracy, local machine learning model to classify
messages and call transcripts as fraud/scam (1) or legitimate/safe (0).
It combines two datasets:
1. 'SMS fraud Dataset.csv' (SMS messages)
2. 'fraud_call.file' (Call transcripts)

Requirements:
    pip install pandas scikit-learn xgboost

To run training locally:
    python train_model.py
================================================================================
"""

import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

# ── Configuration Paths ───────────────────────────────────────────────────────
# Resolve paths relative to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATASET = os.path.join(SCRIPT_DIR, "SMS fraud Dataset.csv")
CALL_DATASET = os.path.join(SCRIPT_DIR, "fraud_call.file")
SCAM_VOICE_DATASET = os.path.join(SCRIPT_DIR, "English_Scam.txt")
NONSCAM_VOICE_DATASET = os.path.join(SCRIPT_DIR, "English_NonScam.txt")
BETTER30_DATASET = os.path.join(SCRIPT_DIR, "BETTER30.csv")
FRAUD_CALLS_DATA_DATASET = os.path.join(SCRIPT_DIR, "fraud_calls_data.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "backend", "models")

def load_data():
    """Load and merge all four datasets, mapping labels to 0 (Safe) and 1 (Scam)."""
    
    # Preprocessing helpers
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Remove bracketed placeholders: e.g. [Greetings], [Company], [You]
        text = re.sub(r'\[[^\]]+\]', ' ', text)
        # Convert to lower case
        text = text.lower()
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_voice_line(line):
        # Remove leading numbering like "1. ", "30. 46. ", "12.\t", etc.
        cleaned = line.strip()
        while True:
            m = re.match(r'^(\d+[\.\t\s]*)+', cleaned)
            if m:
                cleaned = cleaned[m.end():].strip()
            else:
                break
        return clean_text(cleaned)

    # 1. Load SMS Dataset
    if not os.path.exists(CSV_DATASET):
        raise FileNotFoundError(f"Missing required dataset: '{CSV_DATASET}'")
        
    print(f"Reading '{os.path.basename(CSV_DATASET)}'...")
    df_sms = pd.read_csv(CSV_DATASET)
    df_sms = df_sms.rename(columns={"sms": "text", "label": "label"})
    df_sms["text"] = df_sms["text"].apply(clean_text)
    print(f"-> Loaded {len(df_sms)} SMS records.")

    # 2. Load Call Transcript Dataset
    if not os.path.exists(CALL_DATASET):
        raise FileNotFoundError(f"Missing required dataset: '{CALL_DATASET}'")

    print(f"Reading '{os.path.basename(CALL_DATASET)}'...")
    call_texts = []
    call_labels = []
    with open(CALL_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                lbl, text = parts
                cleaned_text = clean_text(text)
                if cleaned_text:
                    call_texts.append(cleaned_text)
                    # Map 'fraud' -> 1, 'normal' -> 0
                    call_labels.append(1 if lbl.strip().lower() == "fraud" else 0)

    df_call = pd.DataFrame({"text": call_texts, "label": call_labels})
    print(f"-> Loaded {len(df_call)} call transcript records.")

    # 3. Load Voice Scam Dataset
    scam_texts = []
    if os.path.exists(SCAM_VOICE_DATASET):
        print(f"Reading '{os.path.basename(SCAM_VOICE_DATASET)}'...")
        with open(SCAM_VOICE_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = clean_voice_line(line)
                if cleaned:
                    scam_texts.append(cleaned)
        print(f"-> Loaded {len(scam_texts)} scam voice call records.")
    else:
        print(f"[WARNING] Optional dataset missing: '{SCAM_VOICE_DATASET}'")

    # 4. Load Voice Non-Scam Dataset
    nonscam_texts = []
    if os.path.exists(NONSCAM_VOICE_DATASET):
        print(f"Reading '{os.path.basename(NONSCAM_VOICE_DATASET)}'...")
        with open(NONSCAM_VOICE_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = clean_voice_line(line)
                if cleaned:
                    nonscam_texts.append(cleaned)
        print(f"-> Loaded {len(nonscam_texts)} non-scam voice call records.")
    else:
        print(f"[WARNING] Optional dataset missing: '{NONSCAM_VOICE_DATASET}'")

    # Build dataframes for voice datasets
    voice_records = []
    for t in scam_texts:
        voice_records.append({"text": t, "label": 1})
    for t in nonscam_texts:
        voice_records.append({"text": t, "label": 0})
        
    df_voice = pd.DataFrame(voice_records)
    print(f"-> Loaded total {len(df_voice)} voice call dataset records.")

    # 5. Load BETTER30 Dataset
    better30_texts = []
    better30_labels = []
    if os.path.exists(BETTER30_DATASET):
        print(f"Reading '{os.path.basename(BETTER30_DATASET)}'...")
        try:
            df_better30 = pd.read_csv(BETTER30_DATASET)
            for idx, row in df_better30.iterrows():
                txt = row.get("TEXT", "")
                lbl = str(row.get("LABEL", "")).strip().lower()
                cleaned_text = clean_text(txt)
                if cleaned_text:
                    better30_texts.append(cleaned_text)
                    if any(s in lbl for s in ["scam", "suspicious", "fraud"]):
                        better30_labels.append(1)
                    else:
                        better30_labels.append(0)
            print(f"-> Loaded {len(better30_texts)} BETTER30 records.")
        except Exception as e:
            print(f"[ERROR] Failed to load BETTER30 dataset: {e}")
    else:
        print(f"[WARNING] BETTER30 dataset missing: '{BETTER30_DATASET}'")

    # 6. Load fraud_calls_data Dataset
    fraud_calls_texts = []
    fraud_calls_labels = []
    if os.path.exists(FRAUD_CALLS_DATA_DATASET):
        print(f"Reading '{os.path.basename(FRAUD_CALLS_DATA_DATASET)}'...")
        try:
            df_fc = pd.read_csv(FRAUD_CALLS_DATA_DATASET, header=None, names=["label", "text"])
            for idx, row in df_fc.iterrows():
                txt = row.get("text", "")
                lbl = str(row.get("label", "")).strip().lower()
                cleaned_text = clean_text(txt)
                if cleaned_text:
                    fraud_calls_texts.append(cleaned_text)
                    if any(s in lbl for s in ["fraud", "scam"]):
                        fraud_calls_labels.append(1)
                    else:
                        fraud_calls_labels.append(0)
            print(f"-> Loaded {len(fraud_calls_texts)} fraud_calls_data records.")
        except Exception as e:
            print(f"[ERROR] Failed to load fraud_calls_data dataset: {e}")
    else:
        print(f"[WARNING] fraud_calls_data dataset missing: '{FRAUD_CALLS_DATA_DATASET}'")

    df_better30_df = pd.DataFrame({"text": better30_texts, "label": better30_labels})
    df_fc_df = pd.DataFrame({"text": fraud_calls_texts, "label": fraud_calls_labels})

    # Combine and clean all dataframes
    df = pd.concat([df_sms, df_call, df_voice, df_better30_df, df_fc_df], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]
    
    print(f"\nTotal combined records: {len(df)}")
    print("Class Distribution:")
    counts = df["label"].value_counts()
    for lbl, cnt in counts.items():
        name = "Scam/Fraud (1)" if lbl == 1 else "Safe/Legitimate (0)"
        print(f"  {name}: {cnt} ({cnt/len(df)*100:.1f}%)")
        
    return df

def train_pipeline():
    # 1. Load data
    try:
        df = load_data()
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        print("Please ensure datasets are in the correct root directory.")
        return

    # 2. Train/Test Split (Stratified to maintain class proportions)
    print("\nSplitting dataset into 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3. Feature Extraction (TF-IDF Vectorizer with bigrams)
    print("Extracting TF-IDF text features (using unigrams and bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=8000, 
        stop_words="english", 
        sublinear_tf=True, 
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train the XGBoost Classifier with tuned hyperparameters for voice/live calls
    print("Training the local XGBoost model...")
    classifier = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        random_state=42,
        eval_metric="logloss"
    )
    classifier.fit(X_train_vec, y_train)

    # 5. Evaluate results
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print(f"EVALUATION METRICS (Accuracy: {accuracy*100:.2f}%)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Scam (1)"]))
    
    # Print Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (Legit classified as Legit): {cm[0][0]}")
    print(f"  False Positives (Legit classified as Scam): {cm[0][1]}")
    print(f"  False Negatives (Scam classified as Legit): {cm[1][0]}")
    print(f"  True Positives (Scam classified as Scam): {cm[1][1]}")
    print("="*60)

    # 6. Save Model Artifacts
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    vec_out = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    model_out = os.path.join(OUTPUT_DIR, "scam_xgb_model.pkl")

    print(f"\nSaving model artifacts to '{OUTPUT_DIR}'...")
    with open(vec_out, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"-> Saved: {vec_out}")

    with open(model_out, "wb") as f:
        pickle.dump(classifier, f)
    print(f"-> Saved: {model_out}")

    print("\nTraining completed successfully! The model is active and integrated.")

if __name__ == "__main__":
    train_pipeline()
