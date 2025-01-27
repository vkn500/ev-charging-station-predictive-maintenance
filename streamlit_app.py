import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load models and preprocessor
maintenance_model = joblib.load('./models/maintenance_model.pkl')
fault_model = joblib.load('./models/fault_model.pkl')
efficiency_model = joblib.load('./models/efficiency_model.pkl')
preprocessor = joblib.load('./models/preprocessor.pkl')

# Set up the app
st.title('EV Charging Station Maintenance Prediction System')

# Input fields
st.header('Input Parameters')
station_id = st.text_input('Station ID')
temperature_c = st.slider('Temperature (°C)', min_value=-30.0, max_value=50.0, step=0.1)
humidity_percent = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
power_usage_kwh = st.slider('Power Usage (kWh)', min_value=0.0, max_value=1000.0, step=1.0)
connection_time_minutes = st.slider('Connection Time (Minutes)', min_value=0, max_value=1440, step=1)
charging_sessions_daily = st.slider('Charging Sessions (Daily)', min_value=0, max_value=500, step=1)
faults_per_week = st.number_input('Faults Per Week', step=1)
downtime_minutes = st.number_input('Downtime (Minutes)', step=1)
maintenance_cost_usd = st.number_input('Maintenance Cost (USD)', step=0.01)
peak_hours_usage_kwh = st.number_input('Peak Hours Usage (kWh)', step=0.1)
non_peak_hours_usage_kwh = st.number_input('Non-Peak Hours Usage (kWh)', step=0.1)
number_of_chargers = st.number_input('Number of Chargers', step=1)
charger_age_years = st.number_input('Charger Age (Years)', step=1)
error_count_monthly = st.number_input('Error Count (Monthly)', step=1)
weather_condition = st.selectbox('Weather Condition', ['Rainy', 'Windy', 'Cloudy', 'Sunny'])
energy_cost_per_kwh = st.number_input('Energy Cost Per kWh', step=0.01)
customer_satisfaction_score = st.number_input('Customer Satisfaction Score', step=0.1)

# Create DataFrame from inputs
data = {
    'Station_ID': [int(station_id) if station_id else 0],
    'Temperature_C': [temperature_c],
    'Humidity_Percent': [humidity_percent],
    'Power_Usage_kWh': [power_usage_kwh],
    'Connection_Time_Minutes': [connection_time_minutes],
    'Charging_Sessions_Daily': [charging_sessions_daily],
    'Faults_Per_Week': [faults_per_week],
    'Downtime_Minutes': [downtime_minutes],
    'Maintenance_Cost_USD': [maintenance_cost_usd],
    'Peak_Hours_Usage_kWh': [peak_hours_usage_kwh],
    'Non_Peak_Hours_Usage_kWh': [non_peak_hours_usage_kwh],
    'Number_of_Chargers': [number_of_chargers],
    'Charger_Age_Years': [charger_age_years],
    'Error_Count_Monthly': [error_count_monthly],
    'Weather_Condition': [weather_condition],
    'Energy_Cost_Per_kWh': [energy_cost_per_kwh],
    'Customer_Satisfaction_Score': [customer_satisfaction_score]
}
df = pd.DataFrame(data)

# One-hot encode the weather condition and preprocess data
if st.button("Predict"):
    df_preprocessed = preprocessor.transform(df)

    # Make predictions
    time_to_next_maintenance = maintenance_model.predict(df_preprocessed)[0]
    fault_probability = fault_model.predict(df_preprocessed)[0]
    usage_efficiency = efficiency_model.predict(df_preprocessed)[0]

    # Display results
    st.header('Predicted Outputs')
    st.write(f"Time to Next Maintenance (Days): {time_to_next_maintenance:.2f}")
    st.write(f"Fault Probability: {fault_probability:.2f}")
    st.write(f"Usage Efficiency (%): {usage_efficiency:.2f}")

    # Visual output
    st.header('Visualization')
    fig, ax = plt.subplots()
    categories = ['Next Maintenance (Days)', 'Fault Probability', 'Usage Efficiency (%)']
    values = [time_to_next_maintenance, fault_probability, usage_efficiency]

    ax.bar(categories, values, color=['coral', 'orchid', 'moccasin'])
    ax.set_ylabel('Values')
    ax.set_title('Predicted Outputs')

    st.pyplot(fig)

 # About section
st.header('About')
st.write('This project aims to enhance the reliability and efficiency of EV charging stations by predicting their maintenance needs. By leveraging machine learning techniques, the system can forecast potential faults and schedule maintenance proactively, thereby minimizing downtime and ensuring a seamless charging experience for users.')

# Developers section
st.header("Developers")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("https://via.placeholder.com/100", caption="Komendra Kumar Dhruvey")
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
