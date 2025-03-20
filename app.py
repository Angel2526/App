import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Custom CSS for light green transparent background
st.markdown(
    """
    <style>
    /* Set background color for the entire app */
    .stApp {
        background-color: rgba(150, 250, 144, 0.3); /* Light green with 30% opacity */
    }

    /* Optional: Style specific components */
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
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset and train the model dynamically
def load_and_train_model(file_path):
    df = pd.read_excel(file_path)
    df.fillna("Unknown", inplace=True)
    
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop(columns=['RESISTANT TO'])  # Change column name if needed
    y = df['RESISTANT TO']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(clf, "antibiotic_resistance_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    
    return clf, label_encoders, X.columns, accuracy

# Load trained model and encoders
def load_model():
    try:
        clf = joblib.load("antibiotic_resistance_model.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        return clf, label_encoders
    except FileNotFoundError:
        return None, None

# Streamlit UI with medical emoji
st.title("MDRO Prediction Tool🔍")
st.markdown("""
    <div class="big-font">
    Enter patient details to predict antibiotic resistance.
    </div>
    """, unsafe_allow_html=True)

# Train model dynamically
file_path = "RE5mon.xlsx"  # Ensure this file is available in the same directory
clf, label_encoders, feature_names, accuracy = load_and_train_model(file_path)
#st.write(f"Model trained with accuracy: **{accuracy * 100:.2f}%**")

# User input form with medical emoji
st.sidebar.header("🩺 Patient Details")
data = {}
for feature in feature_names:
    if feature in label_encoders:
        options = [""] + list(label_encoders[feature].classes_)  # Add empty option
        data[feature] = st.sidebar.selectbox(f"{feature}", options, index=0)  # Default to empty
    else:
        data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)  # Default to 0.0

# Predict button
if st.sidebar.button("Predict Resistance"):
    clf, label_encoders = load_model()
    if clf and label_encoders:
        input_data = []
        for feature in feature_names:
            if feature in label_encoders:
                # Skip if the field is empty
                if data[feature] == "":
                    st.warning(f"Please fill in the field: {feature}")
                    st.stop()
                input_data.append(label_encoders[feature].transform([data[feature]])[0])
            else:
                input_data.append(data[feature])
        
        # Predict the numerical value
        prediction_numeric = clf.predict([input_data])[0]
        
        # Inverse transform to get the original class label
        target_encoder = label_encoders['RESISTANT TO']  # Use the encoder for the target column
        prediction_label = target_encoder.inverse_transform([prediction_numeric])[0]
        
        # Determine if it's High Risk or Low Risk
        num_classes = len(target_encoder.classes_)  # Number of unique classes in the target column
        if num_classes >= 4:
            risk_level = "High Risk"
            risk_color = "red"  # Set font color to red for High Risk
        else:
            risk_level = "Low Risk"
            risk_color = "green"  # Set font color to green for Low Risk
        
        # Display the prediction and risk level with colored font
        st.success(f"Predicted Resistance: **{prediction_label}**")
        st.markdown(f"Risk Level: <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
    else:
        st.error("Model not found. Please retrain it.")