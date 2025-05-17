import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: rgba(150, 250, 144, 0.3);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Train model function
def train_model(file_path):
    df = pd.read_excel(file_path)
    df.fillna("Unknown", inplace=True)

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop(columns=['RESISTANT TO', 'Culture/Organism isolated'])
    y_resistant = df['RESISTANT TO']
    y_organism = df['Culture/Organism isolated']

    X_train, X_test, y_train_r, y_test_r, y_train_o, y_test_o = train_test_split(
        X, y_resistant, y_organism, test_size=0.2, random_state=42
    )

    model_r = RandomForestClassifier(n_estimators=100, random_state=42)
    model_r.fit(X_train, y_train_r)

    model_o = RandomForestClassifier(n_estimators=100, random_state=42)
    model_o.fit(X_train, y_train_o)

    joblib.dump(model_r, "antibiotic_resistance_model.pkl")
    joblib.dump(model_o, "organism_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    return model_r, model_o, label_encoders

# Load models
def load_models():
    try:
        model_r = joblib.load("antibiotic_resistance_model.pkl")
        model_o = joblib.load("organism_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        return model_r, model_o, encoders
    except:
        return None, None, None

# Load and train if not present
file_path = "DATA-ifill.xlsx"
model_r, model_o, label_encoders = train_model(file_path)

# UI Setup
st.title("MDRO Prediction Tool 🔍")
st.markdown("Enter patient details to predict *antibiotic resistance* and *culture/organism isolated*.")

# Sidebar setup
st.sidebar.header("🩺 Patient Details")

# Input fields
mandatory_fields = ["Age", "Gender", "Previous hospitalization within 3 months",
                    "Location of previous admission", "Reason for admission", "Surgery?",
                    "Mechanical Ventilation", "Non-Invasive Ventilation", "Length of Hospital/Critical Care unit stay",
                    "History of Chronic kidney disease/Dialysis", "Severity of Illness",
                    "APACHE II Score", "Neutropenia/immunosuppression"]

comorbidities = ["HYPERTENSION", "T2DM", "CAD", "CKD", "CKD ON HEMODIALYSIS", "DYSLIPIDEMIA",
                 "HYPOTHYROIDISM", "CLD", "AKD", "LRTI", "BPH", "RA", "SEIZURE", "CVA", "COPD",
                 "ASTHMA", "OTHER", "No Comorbidities"]

devices = ["Foleys Catheterization", "Ryles tube", "Endotracheal intubation", "Central line",
           "Blood transfusion", "Dialysis", "Other"]

antibiotics = ["CEFTRIAXONE 1G", "PIPERACILLIN TAZOBACTAM 4.5G", "MEROPENEM 1G", "LEVOFLOXACIN 500MG",
               "METRONIDAZOLE 500MG", "PIPERACILLIN TAZOBACTAM 2.25G", "CEFOPERAZONE SULBACTUM 1.5G",
               "CEFUROXIME 500MG", "AUGMENTIN 1.2G", "CEFTRIAXONE 2G", "MEROPENEM 500MG", "OSELTAMIVIR 75MG",
               "AUGMENTIN 625MG", "CIPROFLOXACIN 500MG", "CEFOPERAZONE SULBACTUM 3G", "ORNIDAZOLE 500MG",
               "CEFTAZIDIME 200MG", "TEICOPLANIN 400MG", "CEFUROXIME AXETIL 500MG", "CEFOPRIM 1.5G",
               "CEFTAZIDIME 1G", "AMPICILLIN CLOXACILLIN 500/500", "RIFAXIMIN 400MG", "FLUCONAZOLE 200MG",
               "SULBACTUM 2G", "LINEZOLID 500MG", "CEFUROXIME 250MG", "CIPROFLOXACIN 250MG", "AMOXICILLIN 1G",
               "AMIKACIN 500MG", "CEFUROXIME 1.5G", "POLYMIXIN B 5LAKH UNITS", "CLINDAMYCIN 600MG",
               "CEFPODIXIME 200MG", "AZTREONAM 1G"]

current_antibiotics = ["PIPERACILLIN TAZOBACTAM 4.5G", "CEFTRIAXONE 1G", "MEROPENEM 1G", "LEVOFLOXACIN 500MG",
                       "METRONIDAZOLE 500MG", "DOXYCYCLINE 100MG", "CEFOPERAZONE SULBACTUM 1.5G",
                       "OSELTAMIVIR 75MG", "CEFUROXIME 1.5G", "CEFTRIAXONE 2G", "AUGMENTIN 1.2G", "CLARITHROMYCIN 500MG"]

# Collect data
data = {}
for field in mandatory_fields:
    if field in label_encoders:
        options = list(label_encoders[field].classes_)

        # Remove 'Nil' from 'Previous hospitalization within 3 months'
        if field == "Previous hospitalization within 3 months" and "Nil" in options:
            options.remove("Nil")

        # Remove duplicates from 'Location of previous admission'
        if field == "Location of previous admission":
            options = sorted(set(options))

        data[field] = st.sidebar.selectbox(f"{field}*", [""] + options)
    else:
        data[field] = st.sidebar.number_input(f"{field}*", value=None, placeholder="Enter a number")

# Antibiotic exposure
st.sidebar.subheader("🧪 Antibiotic Exposure in past 3 months")
antibiotic_exposure = st.sidebar.selectbox("Antibiotic Exposure*", ["", "Nil", "Yes"])
selected_antibiotics = []

if antibiotic_exposure == "Yes":
    selected_antibiotics = st.sidebar.multiselect("Select antibiotics*", antibiotics)
    for ab in antibiotics:
        data[ab] = "Yes" if ab in selected_antibiotics else "No"
else:
    for ab in antibiotics:
        data[ab] = "Nil"

# Add numeric input for number of antibiotics used in the past
data["No. of Antibiotics used in the Past"] = st.sidebar.number_input("No. of Antibiotics used in the Past", min_value=0, step=1)

# Comorbidities
st.sidebar.subheader("🩻 Comorbidities")
selected_comorb = st.sidebar.multiselect("Select", comorbidities)
for com in comorbidities:
    data[com] = "Yes" if com in selected_comorb else "No"

# Devices
st.sidebar.subheader("🩼 Devices Used")
selected_devices = st.sidebar.multiselect("Devices", devices)
for dev in devices:
    data[dev] = "Yes" if dev in selected_devices else "No"

# Current Antibiotics
st.sidebar.subheader("💊 Current Antibiotics")
selected_current = st.sidebar.multiselect("Current Antibiotics", current_antibiotics)
for ab in current_antibiotics:
    data[f"Current: {ab}"] = "Yes" if ab in selected_current else "No"

# Predict button
if st.sidebar.button("Predict Resistance and Culture/Organism"):
    model_r, model_o, label_encoders = load_models()
    if model_r and model_o and label_encoders:
        input_data = []
        for key in data:
            if key in label_encoders:
                try:
                    input_data.append(label_encoders[key].transform([str(data[key])])[0])
                except:
                    input_data.append(label_encoders[key].transform(["Unknown"])[0])
            else:
                try:
                    input_data.append(float(data[key]))
                except:
                    input_data.append(0.0)

        pred_r = model_r.predict([input_data])[0]
        pred_o = model_o.predict([input_data])[0]

        pred_resistant = label_encoders['RESISTANT TO'].inverse_transform([pred_r])[0]
        pred_organism = label_encoders['Culture/Organism isolated'].inverse_transform([pred_o])[0]

        st.success(f"🔬 Predicted Resistance: *{pred_resistant}*")
        st.success(f"🧫 Predicted Organism: *{pred_organism}*")
    else:
        st.error("Model not found. Please retrain.")