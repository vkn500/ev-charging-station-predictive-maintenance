import streamlit as st
import joblib
import numpy as np
import datetime
import plotly

def main():
    st.title("EV Charging Station Maintenance Predictor")

    # Load models and scaler
    model_next_maintenance = joblib.load("./models/model_next_maintenance_days.pkl")
    model_maintenance_needed = joblib.load("./models/model_maintenance_needed.pkl")
    model_fault_probability = joblib.load("./models/model_fault_probability.pkl")
    scaler = joblib.load("./models/scaler.pkl")

    # User inputs
    charging_station_id = st.number_input("Charging Station ID", 0, 5000, value=2500)
    charging_sessions = st.slider("Charging Sessions", 0, 20, value=10)
    total_energy = st.slider("Total Energy Delivered (kW)", 0, 1000, value=500)
    default_date = datetime.date.today() - datetime.timedelta(days=180)
    last_maintenance_date = st.date_input("Last Maintenance Date", value=default_date)
    last_maintenance_days = (datetime.date.today() - last_maintenance_date).days
    charging_duration = st.slider("Charging Duration (Hours)", 0.0, 24.0, value=12.0)

    if st.button("Submit"):
        # Prepare input for scaling
        input_data = np.array([[charging_station_id, charging_sessions,
                                total_energy, last_maintenance_days, charging_duration]])
        input_scaled = scaler.transform(input_data)

        # Predictions
        days_pred = model_next_maintenance.predict(input_scaled)[0]
        maintenance_pred = model_maintenance_needed.predict(input_scaled)[0]
        fault_pred = model_fault_probability.predict(input_scaled)[0]

        # Display results
        st.subheader(f"Station {charging_station_id} Maintenance Report")
        
        maintenance_categories = ['Software Update', 'Connector Cleaning', 'Cable Inspection', 'Battery Check']
        maintenance_category = maintenance_categories[maintenance_pred]

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"Maintenance Needed in {days_pred:.0f} Days:")
            next_maintenance_date = datetime.date.today() + datetime.timedelta(days=days_pred)
            st.subheader(f"{next_maintenance_date}")
        
        with col2:
            st.write("Maintenance Needed:")
            st.subheader(f"{maintenance_category}")
        
        with col3:
            st.write("Fault Probability:")
            st.write(" ")
            st.progress(fault_pred)

        # Plotly visualizations
        import plotly.graph_objs as go

        feature_values = [charging_sessions, total_energy, last_maintenance_days, charging_duration]
        feature_names = ["Charging Sessions", "Total Energy (kW)", "Days Since Last Maintenance", "Charging Duration (Hours)"]

        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

        fig_features = go.Figure([go.Bar(x=feature_names, y=feature_values, marker=dict(color=colors))])
        fig_features.update_layout(title="Feature Overview", xaxis_title="Features", yaxis_title="Values")
        st.plotly_chart(fig_features)

    #About
    st.subheader("About")
    st.write("This project aims to enhance the reliability and efficiency of EV charging stations by predicting their maintenance needs. By leveraging machine learning techniques, the system can forecast potential faults and schedule maintenance proactively, thereby minimizing downtime and ensuring a seamless charging experience for users.")

    #developed by section
    st.subheader("Developed by")
    st.write("Komendra Dhruvey, Shiv Kumar Sao, Tarun Sahu and Vishal Nishad")

if __name__ == '__main__':
    main()