import pickle
import pandas as pd
import streamlit as st

# Set the title of the app
st.title("❤️ Cardio-Vascular Disease Name Prediction ❤️")
st.header("Input Patient's Details")

# Function to load the trained model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict from user input using the loaded model
def predict_disease(model, user_input_df):
    prediction = model.predict(user_input_df)[0]
    return prediction

# Path to the saved model
model_path = 'heart_disease_rf_model.pkl'
model = load_model(model_path)

# Function to collect user inputs
def get_user_input():
    # Input fields for patient details
    Age = st.number_input("Age", min_value=1, max_value=120, value=55)

    col1, col2, col3 = st.columns(3)

    chest_pain = col1.checkbox("Chest Pain?")
    shortness_of_breath = col2.checkbox("Shortness of Breath?")
    fatigue = col3.checkbox("Fatigue?")
    
    Systolic = st.number_input("Systolic Pressure", min_value=1, value=140)
    Diastolic = st.number_input("Diastolic Pressure", min_value=1, value=90)
    Heart_rate = st.number_input("Heart Rate (bpm)", min_value=1, value=100)
    
    lung_sounds = st.checkbox("Lung Sounds?")
    
    cholesterol_level = st.number_input("Cholesterol Level (mg/dL)", min_value=1, value=220)
    ldl_level = st.number_input("LDL Level (mg/dL)", min_value=1, value=150)
    hdl_level = st.number_input("HDL Level (mg/dL)", min_value=1, value=40)
    
    col4, col5, col6 = st.columns(3)
    diabetes = col4.checkbox("Diabetes?")
    atrial_fibrillation = col5.checkbox("Atrial Fibrillation?")
    rheumatic_fever = col6.checkbox("Rheumatic Fever?")
    
    col7, col8, col9 = st.columns(3)
    mitral_stenosis = col7.checkbox("Mitral Stenosis?")
    aortic_stenosis = col8.checkbox("Aortic Stenosis?")
    tricuspid_stenosis = col9.checkbox("Tricuspid Stenosis?")
    
    col10, col11, col12 = st.columns(3)
    pulmonary_stenosis = col10.checkbox("Pulmonary Stenosis?")
    dilated_cardiomyopathy = col11.checkbox("Dilated Cardiomyopathy?")
    hypertrophic_cardiomyopathy = col12.checkbox("Hypertrophic Cardiomyopathy?")
    
    col13, col14, col15 = st.columns(3)
    drug_use = col13.checkbox("Drug Use?")
    fever = col14.checkbox("Fever?")
    chills = col15.checkbox("Chills?")
    
    col16, col17, col18 = st.columns(3)
    alcoholism = col16.checkbox("Alcoholism?")
    hypertension = col17.checkbox("Hypertension?")
    fainting = col18.checkbox("Fainting?")
    
    col19, col20, col21 = st.columns(3)
    dizziness = col19.checkbox("Dizziness?")
    smoking = col20.checkbox("Smoking?")
    obesity = col21.checkbox("Obesity?")
    
    murmur = st.checkbox("Murmur?")

    # Create a DataFrame from the input values
    user_input = {
        'Age': Age,
        'Chest pain': int(chest_pain),
        'Shortness of breath': int(shortness_of_breath),
        'Fatigue': int(fatigue),
        'Systolic': Systolic,
        'Diastolic': Diastolic,
        'Heart rate (bpm)': Heart_rate,
        'Lung sounds': int(lung_sounds),
        'Cholesterol level (mg/dL)': cholesterol_level,
        'LDL level (mg/dL)': ldl_level,
        'HDL level (mg/dL)': hdl_level,
        'Diabetes': int(diabetes),
        'Atrial fibrillation': int(atrial_fibrillation),
        'Rheumatic fever': int(rheumatic_fever),
        'Mitral stenosis': int(mitral_stenosis),
        'Aortic stenosis': int(aortic_stenosis),
        'Tricuspid stenosis': int(tricuspid_stenosis),
        'Pulmonary stenosis': int(pulmonary_stenosis),
        'Dilated cardiomyopathy': int(dilated_cardiomyopathy),
        'Hypertrophic cardiomyopathy': int(hypertrophic_cardiomyopathy),
        'Drug use': int(drug_use),
        'Fever': int(fever),
        'Chills': int(chills),
        'Alcoholism': int(alcoholism),
        'Hypertension': int(hypertension),
        'Fainting': int(fainting),
        'Dizziness': int(dizziness),
        'Smoking': int(smoking),
        'Obesity': int(obesity),
        'Murmur': int(murmur)
    }
    
    return pd.DataFrame(user_input, index=[0])

# Collect user input
user_input_df = get_user_input()

# Predict the outcome
if st.button('Predict'):
    predicted_outcome = predict_disease(model, user_input_df)
    st.subheader(f"Predicted Heart Disease: {predicted_outcome}")
