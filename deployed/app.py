# filepath: /C:/Users/tarun/Downloads/streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Title of the app
st.title('EV Charging Station Maintenance Prediction')

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('../ev_charging_station_maintenance_model.pkl')
    return model

model = load_model()

# Sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    power_consumption = st.sidebar.number_input('Power Consumption (kWh)', min_value=0.0, step=0.1, value=10.0)
    charging_duration = st.sidebar.number_input('Charging Duration (hours)', min_value=0.0, step=0.1, value=2.0)
    charging_rate = st.sidebar.number_input('Charging Rate (kW)', min_value=0.0, step=0.1, value=10.0)
    charging_start_percentage = st.sidebar.number_input('Charging Start Percentage (%)', min_value=0.0, max_value=100.0, step=1.0, value=20.0)
    charging_end_percentage = st.sidebar.number_input('Charging End Percentage (%)', min_value=0.0, max_value=100.0, step=1.0, value=80.0)
    usage_count = st.sidebar.number_input('Usage Count', min_value=0, step=1, value=50)

    data = {
        'power_consumption': power_consumption,
        'charging_duration': charging_duration,
        'charging_rate': charging_rate,
        'charging_start_percentage': charging_start_percentage,
        'charging_end_percentage': charging_end_percentage,
        'usage_count': usage_count
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('Input Features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error('Prediction: Maintenance Needed')
    else:
        st.success('Prediction: No Maintenance Needed')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
