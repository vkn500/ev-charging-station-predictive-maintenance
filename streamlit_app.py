import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def load_models():
    model = joblib.load("./models/random_forest_model.joblib")
    scaler = joblib.load("./models/scaler.joblib")
    return model, scaler

def main():
    st.title("Predictive Maintenance App")
    model, scaler = load_models()

    # Collect user inputs
    charging_sessions = st.number_input("Charging Sessions", min_value=0)
    total_energy_delivered_kW = st.number_input("Total Energy Delivered (kW)", min_value=0.0)
    charging_duration_hours = st.number_input("Charging Duration (hours)", min_value=0.0)
    current = st.number_input("Current (A)", min_value=0.0)
    last_maintenance_date = st.date_input("Last Maintenance Date")
    last_maintenance_time = st.time_input("Last Maintenance Time")
    last_maintenance_date_year = last_maintenance_date.year
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0)
    user_feedback = st.selectbox("User Feedback", ["Positive", "Neutral", "Negative"])
    voltage = st.number_input("Voltage (V)", min_value=0.0)
    location = st.selectbox("Location", ["Location A", "Location B", "Location C", "Location D"])  # example
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    if st.button("Predict"):
        # Convert categorical to numeric (simplified example; adjust as needed)
        location_val = ["Location A", "Location B", "Location C", "Location D"].index(location)
        time_val = 0 if time_of_day == "Morning" else 1 if time_of_day == "Afternoon" else 2 if time_of_day == "Evening" else 3
        user_feedback_val = 0 if user_feedback == "Positive" else 1 if user_feedback == "Neutral" else 2

        # Extract day, month, and hour from date and time inputs
        last_maintenance_date_day = last_maintenance_date.day
        last_maintenance_date_month = last_maintenance_date.month
        last_maintenance_date_hour = last_maintenance_time.hour

        # Build input DataFrame
        input_data = pd.DataFrame([[
            charging_sessions,
            total_energy_delivered_kW,
            charging_duration_hours,
            current,
            last_maintenance_date_day,
            last_maintenance_date_month,
            last_maintenance_date_year,
            last_maintenance_date_hour,
            temperature,
            user_feedback_val,
            voltage,
            location_val,
            time_val
        ]], columns=[
            "charging_sessions",
            "total_energy_delivered_kW",
            "charging_duration_hours",
            "current",
            "last_maintenance_date_day",
            "last_maintenance_date_month",
            "last_maintenance_date_year",
            "last_maintenance_date_hour",
            "temperature",
            "user_feedback",
            "voltage",
            "location",
            "time_of_day"
        ])
        # Ensure the feature names are in the same order as they were during fit
        input_data = input_data[[
            "charging_sessions",
            "total_energy_delivered_kW",
            "charging_duration_hours",
            "current",
            "last_maintenance_date_day",
            "last_maintenance_date_month",
            "last_maintenance_date_year",
            "last_maintenance_date_hour",
            "temperature",
            "user_feedback",
            "voltage",
            "location",
            "time_of_day"
        ]]
        # Scale data
        scaled_data = scaler.transform(input_data)
        # Prediction
        predictions = model.predict(scaled_data)
        next_maintenance_days, maintenance_needed, fault_probability = predictions[0]
        # Display results
        st.write(f"Next Maintenance Days: {next_maintenance_days:.2f}")
        st.write(f"Maintenance Needed (encoded): {maintenance_needed:.2f}")
        st.write(f"Fault Probability: {fault_probability:.2f}")

if __name__ == "__main__":
    main()