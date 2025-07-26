# retrain_model_app.py

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- File Paths ---
DATA_FILE = "user_feedback.csv"
MODEL_FILE = "logreg_mode_predictor.pkl"
ENCODERS_FILE = "label_encoders.pkl"
TARGET_ENCODER_FILE = "target_encoder.pkl"

st.title("🔁 Retrain Mode Preference Model")

# --- Step 1: Check file existence and content ---
if not os.path.exists(DATA_FILE):
    st.warning("⚠️ Data file not found. Submit some preferences first.")
    st.stop()

df = pd.read_csv(DATA_FILE)

if df.empty:
    st.warning("⚠️ No data available in the file. Please submit some data first.")
    st.stop()

# --- Step 2: Encode categorical features ---
cat_cols = [
    "age_group", "gender", "device_usage_frequency", "primary_use",
    "usual_mode", "eye_strain_experience", "mode_choice_factors",
    "daily_screen_time"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- Step 3: Encode target label ---
target_encoder = LabelEncoder()
df["mode_preference"] = target_encoder.fit_transform(df["mode_preference"])

# --- Step 4: Check class diversity ---
if len(df["mode_preference"].unique()) < 2:
    st.error("❌ Not enough class diversity to train. At least two different mode preferences are required.")
    st.stop()

# --- Step 5: Feature engineering ---
df["comfort_diff"] = df["comfort_dark"] - df["comfort_light"]
df["focus_diff"] = df["focus_dark"] - df["focus_light"]
df["eye_strain_diff"] = df["eye_strain_dark"] - df["eye_strain_light"]

X = df.drop(columns=["mode_preference"])
y = df["mode_preference"]

# --- Step 6: Train model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# --- Step 7: Save model and encoders ---
joblib.dump(model, MODEL_FILE)
joblib.dump(encoders, ENCODERS_FILE)
joblib.dump(target_encoder, TARGET_ENCODER_FILE)

st.success("✅ Model retrained and saved successfully!")
st.markdown("- `logreg_mode_predictor.pkl` updated ✅\n- `label_encoders.pkl` updated ✅\n- `target_encoder.pkl` updated ✅")
