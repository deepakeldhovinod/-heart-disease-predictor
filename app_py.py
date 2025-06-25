import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load model, scaler, and features
model = load_model("heart_disease_bestmodel.keras")
scaler = load("scaler.joblib")
features = load("features.joblib")

# Page config and styling
st.set_page_config(page_title="Heart Health Check", page_icon="💓", layout="centered")
st.markdown("""
    <style>
      body {
            background-color: #e6f2ff;
        }
        .main {
            background-color: #000000;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #003366;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 10px;
            padding: 0.5em 2em;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966489.png", width=80)
st.sidebar.markdown('<h1 style="color:lightblue;">🏥 About</h1>', unsafe_allow_html=True)
st.sidebar.info("""
This app predicts the **risk of heart disease** using health data.

:mag: Fill in your details on the right  
:brain: Uses a trained machine learning model  
:chart_with_upwards_trend: Gives a probability and prediction
""")

st.markdown('<h1 style="color:lightblue;">💓 Heart Disease Prediction App</h1>', unsafe_allow_html=True)

st.markdown('<h4 style="color:lightblue;">Please enter your health information below:</h4>',unsafe_allow_html=True)

# Desired and binary/categorical features
desired_features = [
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'Agegroup',
    'BMXBMI', 'BMI_category', 'BPXSY1', 'BPXDI1', 'Hypertension_Stage',
    'Smoker', 'Alcoholic', 'Poor_Sleep',
    'Medication_Count',
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TFIBE', 'DR1TSFAT',
    'DR1TCHOL', 'DR1TSUGR', 'DR1TSODI',
    'PAQ605', 'PAQ620', 'PAD615', 'PAD630'
]

FEATURES = [feat for feat in features if any(feat.startswith(d) for d in desired_features)]

raw_input = {}
binary_features = ['Smoker', 'Alcoholic', 'Poor_Sleep', 'PAQ605', 'PAQ620']
categorical_labels = {
    'BMI_category': {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3},
    'Hypertension_Stage': {'Normal': 0, 'Elevated': 1, 'Stage 1': 2, 'Stage 2': 3, 'Hypertensive Crisis': 4},
    'RIDRETH1': {'Mexican American': 1, 'Other Hispanic': 2, 'Non-Hispanic White': 3, 'Non-Hispanic Black': 4, 'Other Race - Including Multi-Racial': 5},
    'Agegroup': {'infant': 0, 'child': 1, 'youth': 2, 'elder': 3, 'senior': 4}
}

column_labels = {
    'RIDAGEYR': "Age (in years)", 'RIAGENDR': "Gender", 'RIDRETH1': "Ethnicity", 'Agegroup': "Age Group (Custom Category)",
    'BMXBMI': "BMI", 'BMI_category': "BMI Category", 'BPXSY1': "Systolic Blood Pressure", 'BPXDI1': "Diastolic Blood Pressure",
    'Hypertension_Stage': "Hypertension Stage", 'Smoker': "Are you a smoker?", 'Alcoholic': "Do you drink alcohol?",
    'Poor_Sleep': "Do you sleep poorly?", 'Medication_Count': "Number of Medications Taken",
    'DR1TKCAL': "Energy Intake (kcal)", 'DR1TPROT': "Protein Intake (g)", 'DR1TCARB': "Carbohydrate Intake (g)",
    'DR1TFIBE': "Fiber Intake (g)", 'DR1TSFAT': "Saturated Fat Intake (g)", 'DR1TCHOL': "Cholesterol Intake (mg)",
    'DR1TSUGR': "Sugar Intake (g)", 'DR1TSODI': "Sodium Intake (mg)",
    'PAQ605': "Do you do moderate activity?", 'PAQ620': "Do you do vigorous activity?",
    'PAD615': "Minutes of moderate activity per day", 'PAD630': "Minutes of vigorous activity per day"
}

# Defaults for normal male
default_values = {'RIDAGEYR': 30, 'RIAGENDR': 1, 'RIDRETH1': 3, 'Agegroup': 2, 'BMXBMI': 24.5, 'BMI_category': 1,
    'BPXSY1': 115, 'BPXDI1': 75, 'Hypertension_Stage': 0, 'Smoker': 0, 'Alcoholic': 0, 'Poor_Sleep': 0,
    'Medication_Count': 0, 'DR1TKCAL': 2200, 'DR1TPROT': 80, 'DR1TCARB': 250, 'DR1TFIBE': 25, 'DR1TSFAT': 20,
    'DR1TCHOL': 180, 'DR1TSUGR': 50, 'DR1TSODI': 1800, 'PAQ605': 1, 'PAQ620': 1, 'PAD615': 30, 'PAD630': 20}

handled_bases = set()
for feature in FEATURES:
    base = next((d for d in desired_features if feature.startswith(d)), None)
    if base in handled_bases:
        continue
    handled_bases.add(base)
    label = column_labels.get(base, base)
    default = default_values.get(base, 0.0)

    if base == 'RIAGENDR':
        user_value = st.selectbox(label, ["Male", "Female"], index=0 if default == 1 else 1, key=base)
        raw_input[base] = 1 if user_value == "Male" else 0

    elif base in binary_features:
        user_value = st.selectbox(label, ["Yes", "No"], index=0 if default == 1 else 1, key=base)
        raw_input[base] = 1 if user_value == "Yes" else 0

    elif base in categorical_labels:
        options = list(categorical_labels[base].keys())
        reverse_map = {v: k for k, v in categorical_labels[base].items()}
        user_choice = st.selectbox(label, options, index=options.index(reverse_map.get(default, options[0])), key=base)
        raw_input[base] = categorical_labels[base][user_choice]

    else:
        raw_input[base] = st.number_input(label=label, value=default, key=base)

if st.button(":brain: Predict"):
    input_df = pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df)
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURES]
    input_scaled = scaler.transform(input_df)
    prob = model.predict(input_scaled)[0][0]
    pred = int(prob > 0.5)

    st.markdown("---")
    st.subheader(":stethoscope: Result:")
    if pred:
        st.error("🔴 **High Risk of Heart Disease**")
        st.markdown("You may be at a higher risk based on your health data. Please consult a healthcare professional for personalized advice.")
    else:
        st.success("🟩 **Low Risk of Heart Disease**")
        st.markdown("Based on your inputs, you are currently at a low risk of heart disease. Continue maintaining a healthy lifestyle.")

    st.markdown(f":bar_chart: **Model Confidence:** `{prob:.2%}`")

    st.markdown("#### :mag: Explanation (Top 10 Features by SHAP Value)")
    # Create a dummy background of 100 rows with zeros (or representative inputs if available)
    background_data = np.zeros((100, input_scaled.shape[1]))  # shape: (100, num_features)

    # Create SHAP explainer with correct background
    explainer = shap.DeepExplainer(model, background_data)

    # Run explanation
    shap_values = explainer(input_scaled)


    shap_df = pd.DataFrame(shap_values.values[0], index=input_df.columns, columns=['SHAP Value'])
    shap_df = shap_df[shap_df['SHAP Value'] != 0]
    top_shap = shap_df[shap_df['SHAP Value'] != 0].abs().sort_values(by='SHAP Value', ascending=False).head(10)


    if not top_shap.empty:
      fig, ax = plt.subplots()
      top_shap['SHAP Value'].plot(kind='barh', color='coral', ax=ax)
      ax.set_title("Top 10 Contributing Features (SHAP Values)")
      ax.invert_yaxis()
      st.pyplot(fig)
    else:
      st.info("SHAP values are all zero for this input. Try changing the values to see feature contributions.")

