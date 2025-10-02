
import streamlit as st
import pandas as pd
from predictor import FetalHealthPredictor

st.set_page_config(page_title="Fetal Health Predictor")
st.title('👶 Predictor Web App')
st.markdown("A project for the MLDA@EEE Datathon 2025 to predict fetal health from CTG data.")

predictor = FetalHealthPredictor()

st.sidebar.header('Enter the 10 Core Metrics')
st.sidebar.markdown("Drag the sliders to input the CTG analysis values:")

input_data = {
    'LB': st.sidebar.slider('Baseline Heart Rate (LB)', 80, 200, 120),
    'AC': st.sidebar.slider('Accelerations (AC)', 0, 30, 1),
    'FM': st.sidebar.slider('Fetal Movements (FM)', 0, 100, 0),
    'UC': st.sidebar.slider('Uterine Contractions (UC)', 0, 30, 3),
    'DS': st.sidebar.slider('Severe Decelerations (DS)', 0, 10, 0),
    'DP': st.sidebar.slider('Prolonged Decelerations (DP)', 0, 10, 0),
    'ASTV': st.sidebar.slider('Abnormal Short Term Variability (ASTV)', 10, 90, 47),
    'MSTV': st.sidebar.slider('Mean Short Term Variability (MSTV)', 0.0, 7.0, 1.2, step=0.1),
    'ALTV': st.sidebar.slider('Abnormal Long Term Variability (ALTV)', 0, 60, 10),
    'Nzeros': st.sidebar.slider('Histogram Zeros (Nzeros)', 0, 20, 0)
}

st.header('Result')

if st.button('Analyze'):
    with st.spinner('~Please wait'):
        result = predictor.predict(input_data)
        
        prediction_code = result.get("PREDICTION")
        
        inverse_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
        prediction_label = inverse_mapping.get(prediction_code, "Unknown")

        if prediction_code == 1:
            st.success(f"### Prediction: {prediction_label}")
            st.write("The assessment indicates the fetal health status is **Normal**.")
        elif prediction_code == 2:
            st.warning(f"### Prediction: {prediction_label}")
            st.write("The assessment indicates the fetal health status is **Suspect**. Further observation by a clinician is recommended.")
        else:
            st.error(f"### Prediction: {prediction_label}")
            st.write("The assessment indicates the fetal health status is **Pathologic**. Immediate attention by a clinician is advised.")
