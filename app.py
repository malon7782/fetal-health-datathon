import streamlit as st
import pandas as pd
from src.predictor import FetalHealthPredictor 

st.set_page_config(page_title="fetal-health-predictor", layout="wide")
st.title('👶')
st.markdown("A project for the MLDA@EEE Datathon 2025 to predict fetal health from CTG data.")

st.sidebar.header('values')

lb = st.sidebar.slider('(LB)', 80, 200, 120)
ac = st.sidebar.slider('(AC)', 0, 30, 1)
fm = st.sidebar.slider('(FM)', 0, 100, 0)
uc = st.sidebar.slider('(UC)', 0, 30, 3)
dl = st.sidebar.slider('(DL)', 0, 30, 2)
ds = st.sidebar.slider('(DS)', 0, 10, 0)
dp = st.sidebar.slider('(DP)', 0, 10, 0)
astv = st.sidebar.slider('(ASTV)', 10, 90, 45)

st.header('Results')

result_area = st.empty()




@st.cache_resource
def load_predictor():
    predictor = FetalHealthPredictor(model_path='fake_path.joblib', scaler_path='fake_path.joblib')
    return predictor

predictor = load_predictor()


st.header('Results')
result_area = st.empty()
result_area.info("")

if st.button('Start'):
    if predictor:
        input_data = {
            'LB': lb, 'AC': ac, 'FM': fm, 'UC': uc, 'DL': dl, 
            'DS': ds, 'DP': dp, 'ASTV': astv,
            'ALTV': 43, 'Width': 100, 'Min': 90, 'Max': 190, 
            'Nmax': 10, 'Nzeros': 1, 'Mode': 120, 'Mean': 122, 
            'Median': 123, 'Variance': 10, 'Tendency': 0
        }

        with st.spinner('analysing'):
            result = predictor.predict(input_data)
        
        result_area.empty() 
        if result.get("prediction_label") == 'Normal':
            st.success(f"{result['prediction_label']}")
        elif result.get("prediction_label") == 'Suspect':
            st.warning(f"{result['prediction_label']}")
        else:
            st.error(f"{result['prediction_label']}")
        
        st.json(result['confidence'])
