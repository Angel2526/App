import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please train and place it in the same directory.")
    st.stop()

st.set_page_config(page_title="Clinical Data Predictor", layout="wide")

# Center the title
st.markdown(
    "<h1 style='text-align: center; color: white;'>MDRO Prediction Tool 🔍</h1>",
    unsafe_allow_html=True
)

# === Basic Details ===
st.markdown("<h4 style='color:white;'>👤 Basic Details</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Enter age")
with col2:
    gender = st.selectbox("Gender", ["Select", "Male", "Female"])
    if gender == "Select":
        gender = ""

# === Comorbidities ===
with st.expander("🩺 Comorbidities", expanded=False):
    comorbidity_labels = [
        "HYPERTENSION", "T2DM", "CAD", "CKD", "CKD ON HEMODIALYSIS", "DYSLIPIDEMIA",
        "HYPOTHYROIDISM", "CLD", "AKD", "LRTI", "BPH", "RA", "SEIZURE",
        "CVA", "COPD", "ASTHMA", "OTHER", "No Comorbidities"
    ]
    comorbidity_inputs = {}
    cols = st.columns(3)
    for idx, label in enumerate(comorbidity_labels):
        with cols[idx % 3]:
            comorbidity_inputs[label] = st.checkbox(label, key=f"comorb_{label}")

# === Hospitalization and Admission ===
with st.expander("🏥 Hospitalization Details", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        prev_hosp = st.selectbox("Previous hospitalization within 3 months", ["Select", "Yes", "No"])
        surgery = st.selectbox("Surgery?", ["Select", "Yes", "No"])
    with col2:
        loc_options = ["Select", "Hospital ward", "From other hospital", "Emergency", "Intermediate care unit"]
        selected_loc = st.selectbox("Location of previous admission", loc_options)
        prev_loc = st.text_input("Specify location") if selected_loc == "Other" else selected_loc

        reason_options = ["Select", "Medical emergency", "Surgical emergency"]
        selected_reason = st.selectbox("Reason for admission", reason_options)
        admission_reason = st.text_input("Specify reason") if selected_reason == "Other" else selected_reason

# === Respiratory Support ===
with st.expander("🫁 Respiratory Support", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        mech_vent = st.selectbox("Mechanical Ventilation", ["Select", "Yes", "No"])
    with col2:
        niv = st.selectbox("Non-Invasive Ventilation", ["Select", "Yes", "No"])

# === Hospital Stay ===
with st.expander("🛏️ Hospital Stay", expanded=False):
    stay_length = st.number_input("Length of Hospital/Critical Care unit stay (days)", min_value=0, value=None, placeholder="Enter number of days")

# === Kidney History ===
with st.expander("🧪 Kidney History", expanded=False):
    kidney_history = st.selectbox("Chronic Kidney Disease / Dialysis", ["Select", "Yes", "No"])

# === Devices Used ===
with st.expander("🩻 Devices Used", expanded=False):
    device_labels = ["Foleys Catheterization", "Ryles tube", "Endotracheal intubation", "Central line", "Blood transfusion", "Dialysis"]
    devices_used = {}
    cols = st.columns(3)
    for idx, label in enumerate(device_labels):
        with cols[idx % 3]:
            devices_used[label] = st.checkbox(label, key=f"device_{label}")

# === Past Antibiotic Exposure ===
with st.expander("💊 Past Antibiotic Exposure (Last 3 Months)", expanded=False):
    antibiotic_past = st.selectbox("Was there any antibiotic exposure?", ["Select", "Yes", "No"])
    past_antibiotic_inputs = {}
    past_antibiotics_labels = [
        "CEFTRIAXONE 1G", "PIPERACILLIN TAZOBACTAM 4.5G", "MEROPENEM 1G", "LEVOFLOXACIN 500MG",
        "METRONIDAZOLE 500MG", "PIPERACILLIN TAZOBACTAM 2.25G", "CEFOPERAZONE SULBACTUM 1.5G",
        "CEFUROXIME 500MG", "AUGMENTIN 1.2G", "CEFTRIAXONE 2G", "MEROPENEM 500MG", "OSELTAMIVIR 75MG",
        "AUGMENTIN 625MG", "CIPROFLOXACIN 500MG", "CEFOPERAZONE SULBACTUM 3G", "ORNIDAZOLE 500MG",
        "CEFTAZIDIME 200MG", "TEICOPLANIN 400MG", "CEFUROXIME AXETIL 500MG", "CEFOPRIM 1.5G",
        "CEFTAZIDIME 1G", "AMPICILLIN CLOXACILLIN 500/500", "RIFAXIMIN 400MG", "FLUCONAZOLE 200MG",
        "SULBACTUM 2G", "LINEZOLID 500MG", "CEFUROXIME 250MG", "CIPROFLOXACIN 250MG", "AMOXICILLIN 1G",
        "AMIKACIN 500MG", "CEFUROXIME 1.5G", "POLYMIXIN B 5LAKH UNITS", "CLINDAMYCIN 600MG",
        "CEFPODIXIME 200MG", "AZTREONAM 1G"
    ]
    if antibiotic_past == "Yes":
        st.markdown("Select antibiotics administered in the last 3 months:")
        cols = st.columns(3)
        for idx, label in enumerate(past_antibiotics_labels):
            with cols[idx % 3]:
                past_antibiotic_inputs[label] = st.checkbox(label, key=f"past_abx_{label}")

# === Current Antibiotics ===
with st.expander("🧬 Current Antibiotics", expanded=False):
    current_antibiotics = [
        "PIPERACILLIN TAZOBACTAM 4.5G", "CEFTRIAXONE 1G", "MEROPENEM 1G", "LEVOFLOXACIN 500MG", "METRONIDAZOLE 500MG",
        "DOXYCYCLINE 100MG", "CEFOPERAZONE SULBACTUM 1.5G", "OSELTAMIVIR 75MG", "CEFUROXIME 1.5G", "CEFTRIAXONE 2G",
        "AUGMENTIN 1.2G", "CLARITHROMYCIN 500MG", "PIPERACILLIN TAZOBACTAM 2.25G", "CLINDAMYCIN 600MG",
        "MEROPENEM 500MG", "LINEZOLID 600MG", "POLYMIXIN B 5LAKH UNITS", "CEFOPERAZONE SULBACTUM 3G", "VANCOMYCIN 1G",
        "FLUCONAZOLE 200MG", "CEFTAZIDIME 1G", "CIPROFLOXACIN 400MG", "VORICONAZOLE 200MG", "FLUCONAZOLE 150MG",
        "CEFUROXIME 500MG", "AMPICILLIN CLOXACILLIN 1G", "ORNIDAZOLE 500MG", "CEFTAZIDIME 1.25G", "TEICOPLANIN 400MG",
        "AZITHROMYCIN 500MG", "CLARITHROMYCIN 250MG", "CIPROFLOXACIN 200MG", "MINOCYCLINE 100MG", "AZTREONAM 1G",
        "AMIKACIN 500MG", "Other Antibiotics", "No Antibiotics Administered"
    ]
    current_antibiotic_inputs = {}
    cols = st.columns(3)
    for idx, label in enumerate(current_antibiotics):
        with cols[idx % 3]:
            current_antibiotic_inputs[label] = st.checkbox(label, key=f"current_abx_{label}")

# === Organism Isolated ===
with st.expander("🦠 Culture/Organism Isolated", expanded=False):
    organism_choices = sorted([
        "Blood culture - E.coli", "Urine culture - E.coli", "Urine culture - Klebsiella pneumoniae",
        "Endotracheal secretion - Klebsiella pneumoniae", "Pus culture - Klebsiella pneumoniae",
        "Pus culture - E.coli", "Endotracheal secretion - Candida albicans", "Blood culture - Acinetobacter baumanii",
        "Sputum culture - Burkholderia cepacia", "Tissue culture - Cornbacterium striatum",
        "Blood culture - Klebsiella pneumoniae", "Urine culture - Acenitobacter baumanii",
        "Blood culture - Coagulase negative staphylococci", "Blood culture - Pseudomonas aeruginosa",
        "Endotracheal secretion - Burkholderia cepacia", "Blood culture - Gram negative bacilli",
        "Endotracheal secretion - Acinetobacter iwoffii", "Blood culture - Coagulase negative Staphylococci",
        "Tracheostomy secretion - Klebsiella pneumoniae", "Pus culture - Pseudomonas aeruginosa",
        "Sputum culture - Klebsiella pneumoniae", "Blood culture - Staphylococcus aureus",
        "Pus culture - Methicillin Resistant Staphylococcus aureus", "Blood culture - Candida tropicalis",
        "Pus culture - Enterococcus faecium", "Blood culture - Streptococcus species",
        "Plueral fluid - Candida species", "Sputum culture - E.coli", "Pus culture - E. Coli",
        "Endotracheal secretion - Candida tropicalis", "Endotracheal secretion - Pseudomonas aeruginosa",
        "Blood culture - Salmonella typhimurium", "Sputum sample - H3N2 VIRUS DETECTED",
        "Tissue culture - Acinetobacter baumanii", "Pus culture - Cornybacterium striatum",
        "Endotracheal secretion - Acenitobacter baumanii", "Endotracheal secretion - Acinetobacter baumannii",
        "Sputum culture - Enterobacter cloacea", "Urine culture - Heavy mixed bacterial growth",
        "Tissue culture - Pseudomonas aeruginosa", "Sputum culture - Candida albicans",
        "Sputum culture - Fungus, few budding yeast cells with pseudohyphae",
        "Sputum culture - moderate gram +ve bacilli, few gram positive cocci, budding yeast like cell with pseudohyphae",
        "Pus culture - Streptococcus pyogenes", "Blood culture - Staphylococcus epidermidis",
        "Urine culture - Enterococcus faecium", "Blood culture - Enterococcus faecalis",
        "Urine culture - Enterococcus species and also No"
    ])
    selected_organism = st.selectbox("Organism Isolated", ["Select"] + organism_choices)
    if selected_organism == "Select":
        selected_organism = ""

# === Prediction ===
if st.button("Predict"):
    if age is None:
        st.error("Please enter the patient's age.")
    elif not gender:
        st.error("Please select the patient's gender.")
    elif prev_hosp == "Select" or surgery == "Select":
        st.error("Please select options for previous hospitalization and surgery.")
    elif selected_loc == "Select":
        st.error("Please select the location of previous admission.")
    elif selected_reason == "Select":
        st.error("Please select the reason for admission.")
    elif mech_vent == "Select" or niv == "Select":
        st.error("Please select respiratory support options.")
    elif stay_length is None:
        st.error("Please enter the length of stay.")
    elif kidney_history == "Select":
        st.error("Please select kidney history.")
    elif antibiotic_past == "Select":
        st.error("Please select past antibiotic exposure.")
    elif not selected_organism:
        st.error("Please select the organism isolated.")
    else:
        input_data = {
            "Age": age,
            "Gender": gender,
            **comorbidity_inputs,
            "Previous hospitalization within 3 months": prev_hosp,
            "Location of previous admission": prev_loc,
            "Reason for admission": admission_reason,
            "Surgery?": surgery,
            "Mechanical Ventilation": mech_vent,
            "Non-Invasive Ventilation": niv,
            "Length of Hospital/Critical Care unit stay": stay_length,
            "History of Chronic kidney disease/Dialysis": kidney_history,
            **devices_used,
            "Antibiotic exposure in the past 3 months": antibiotic_past,
            **past_antibiotic_inputs,
            **current_antibiotic_inputs,  # ✅ fixed here
            "Culture/Organism Isolated": selected_organism,
        }

        df_input = pd.DataFrame([input_data])

        try:
            expected_cols = model.feature_names_in_
            df_input = pd.get_dummies(df_input)
            for col in expected_cols:
                if col not in df_input:
                    df_input[col] = 0
            df_input = df_input[expected_cols]

            prediction = model.predict(df_input)
            organism = prediction[0][0]
            resistance_str = prediction[0][1]

            if resistance_str and resistance_str.strip().lower() not in ["", "none", "not resistant"]:
                resistant_list = [x.strip() for x in resistance_str.split(",") if x.strip()]
                risk_level = "High Risk" if len(resistant_list) > 3 else "Low Risk"
            else:
                risk_level = "Low Risk"

            st.success("Prediction Results")
            st.markdown(f"""  
**Resistant To**: `{resistance_str}`  
**Risk Level**: **{risk_level}**
""")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
