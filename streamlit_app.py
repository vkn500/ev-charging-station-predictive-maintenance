import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import plotly.express as px  # Add this import

def load_models():
    model = joblib.load("./models/random_forest_model.joblib")
    scaler = joblib.load("./models/scaler.joblib")
    return model, scaler

# Load maintenance_needed.txt to get maintenance categories
maintenance_needed_path = Path("./notebooks/maintenance_needed.txt")
with maintenance_needed_path.open("r") as f:
    maintenance_classes = sorted(set(line.strip() for line in f if line.strip()))
le = LabelEncoder()
le.classes_ = np.array(maintenance_classes)

def main():
    st.title("Predictive Maintenance for EV Charging Stations")
    model, scaler = load_models()

    # Collect user inputs
    charging_sessions = st.slider("Charging Sessions", min_value=0, max_value=30, value=15)
    total_energy_delivered_kW = st.slider("Total Energy Delivered (kW)", min_value=0.0, max_value=1000.0, value=500.0)
    charging_duration_hours = st.slider("Charging Duration (hours)", min_value=0.0, max_value=24.0, value=12.0)
    current = st.slider("Current (A)", min_value=0.0, max_value=110.0, value=55.0)
    last_maintenance_date = st.date_input("Last Maintenance Date")
    last_maintenance_time = st.time_input("Last Maintenance Time")
    last_maintenance_date_year = last_maintenance_date.year
    temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    user_feedback_val = st.slider("Average User Feedback", min_value=1, max_value=5, value=3)
    voltage = st.slider("Voltage (V)", min_value=150, max_value=450, value=300)
    location = st.selectbox("Location", ["Location A", "Location B", "Location C", "Location D"])  # example
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    if st.button("Predict"):
        # Convert categorical to numeric (simplified example; adjust as needed)
        location_val = ["Location A", "Location B", "Location C", "Location D"].index(location)
        time_val = 0 if time_of_day == "Morning" else 1 if time_of_day == "Afternoon" else 2 if time_of_day == "Evening" else 3

        # Extract day, month, and hour from date and time inputs
        last_maintenance_date_year = last_maintenance_date.year
        last_maintenance_date_month = last_maintenance_date.month
        last_maintenance_date_day = last_maintenance_date.day
        last_maintenance_date_hour = last_maintenance_time.hour

        # Build input DataFrame with datetime split
        input_data = pd.DataFrame([[
            location_val,
            time_val,
            charging_sessions,
            total_energy_delivered_kW,
            charging_duration_hours,
            temperature,
            voltage,
            current,
            user_feedback_val,
            last_maintenance_date_year,
            last_maintenance_date_month,
            last_maintenance_date_day,
            last_maintenance_date_hour
        ]], columns=[
            'location',
            'time_of_day',
            'charging_sessions',
            'total_energy_delivered_kW',
            'charging_duration_hours',
            'temperature',
            'voltage',
            'current',
            'user_feedback',
            'last_maintenance_date_year',
            'last_maintenance_date_month',
            'last_maintenance_date_day',
            'last_maintenance_date_hour'
        ])
        # Ensure the feature names are in the same order as they were during fit
        input_data = input_data[[
            'location',
            'time_of_day',
            'charging_sessions',
            'total_energy_delivered_kW',
            'charging_duration_hours',
            'temperature',
            'voltage',
            'current',
            'user_feedback',
            'last_maintenance_date_year',
            'last_maintenance_date_month',
            'last_maintenance_date_day',
            'last_maintenance_date_hour'
        ]]
        
        # Make prediction
        prediction = model.predict(scaler.transform(input_data))
        maintenance_needed = prediction[:, 0].astype(int).flatten()
        next_maintenance_days = prediction[:, 1].flatten()[0]
        fault_probability = prediction[:, 2].flatten()[0]

        # Ensure prediction labels are within known classes
        if np.any(np.isin(maintenance_needed, le.classes_)):
            maintenance_needed_cat = le.inverse_transform(maintenance_needed)[0]
        else:
            maintenance_needed_cat = "Unknown"

        # Visualize the output
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Maintenance Needed", maintenance_needed_cat)
        col2.metric("Next Maintenance in Days", next_maintenance_days)
        col3.progress(int(fault_probability * 100), "Fault Probability")
        
        # Additional Visualizations
        st.subheader(" ")

        st.subheader("Additional Insights")
        maintenance_counts = pd.Series(maintenance_classes).value_counts()
        fig_pie = px.pie(values=maintenance_counts.values, names=maintenance_counts.index, title="Maintenance Category Distribution")
        st.plotly_chart(fig_pie)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_line = px.line(x=input_data['temperature'], y=input_data['voltage'],
               labels={'x': 'Temperature (°C)', 'y': 'Voltage (V)'},
               title="Temperature vs. Voltage")
            st.plotly_chart(fig_line)
        
        with col2:
            fig_scatter = px.scatter(x=input_data['charging_sessions'], y=input_data['total_energy_delivered_kW'],
                 labels={'x': 'Charging Sessions', 'y': 'Total Energy Delivered (kW)'},
                 title="Energy Delivered vs. Charging Sessions")
            st.plotly_chart(fig_scatter)

    # About section
    st.header('About')
    st.write('This project aims to enhance the reliability and efficiency of EV charging stations by predicting their maintenance needs. By leveraging machine learning techniques, the system can forecast potential faults and schedule maintenance proactively, thereby minimizing downtime and ensuring a seamless charging experience for users.')

    # Developers section
    st.header("Developers")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("./img/komendra.png", caption="Komendra Kumar Dhruvey")
        st.markdown("GitHub: [@komendra](#)  \nLinkedIn: [Profile](#)")

    with col2:
        st.image("https://via.placeholder.com/100", caption="Shiv Kumar Sao")
        st.markdown("GitHub: [@shivkumar](#)  \nLinkedIn: [Profile](#)")

    with col3:
        st.image("./img/tarun.png", caption="Tarun Kumar Sahu")
        st.markdown("GitHub: [badflametarun](https://github.com/badflametarun)  \nLinkedIn: [Connect](https://linkedin.com/in/gargitarun)")

    with col4:
        st.image("https://via.placeholder.com/100", caption="Vishal Nishad")
        st.markdown("GitHub: [@vishal](#)  \nLinkedIn: [Profile](#)")

if __name__ == "__main__":
    main()