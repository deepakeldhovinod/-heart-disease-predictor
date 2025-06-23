
import streamlit as st
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt

# Load model, scaler, and training feature list
model = load_model("heart_disease_bestmodel.keras")
scaler = load("scaler.joblib")
FEATURES = load("features.joblib")

st.title("💓 Heart Disease Prediction App")

st.markdown("Please enter your health information below:")

# Create user input fields dynamically
raw_input = {}
for feature in FEATURES:
    base = feature.split('_')[0]  # base name before one-hot suffix
    if base not in raw_input:
        raw_input[base] = st.number_input(f"{base}", value=0.0)

# Prepare prediction when button is clicked
if st.button("Predict"):
    input_df = pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df)

    # Ensure all expected dummy columns are present
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training feature order
    input_df = input_df[FEATURES]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict(input_scaled)[0][0]
    pred = int(prob > 0.5)

    st.markdown(f"🧠 **Prediction:** {'🟥 Positive for Heart Disease' if pred else '🟩 Negative'}")
    st.markdown(f"📊 **Probability:** `{prob:.3f}`")

    # SHAP explanation
    explainer = shap.Explainer(model, input_scaled)
    shap_values = explainer(input_scaled)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("### 🔍 SHAP Feature Impact")
    shap.summary_plot(shap_values.values, input_df, plot_type='bar')
    st.pyplot()
