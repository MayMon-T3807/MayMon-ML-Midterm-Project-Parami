# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("✈️ Flight Delay Predictor")
st.markdown("Predict if your flight will be delayed by more than 15 minutes")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This model predicts flight delays (>15 minutes) using:
    - Airline and airport information
    - Date and time of flight  
    - Flight duration and distance
    - Derived features (rush hour, season, etc.)
    """)
    
    # Check if model file exists
    if os.path.exists('flight_delay2.pkl'):
        file_size = os.path.getsize('flight_delay2.pkl') / (1024 * 1024)
        st.success(f"✅ Model file found: {file_size:.1f} MB")
    else:
        st.error("❌ model not found!")
        st.info("Please upload model_protocol4.pkl to the same directory")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('flight_delay2.pkl')
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {str(e)[:100]}")
        return None

# Load the model
model = load_model()

# Default airline and airport data (from your notebook)
airline_mapping = {
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines', 
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
    'HA': 'Hawaiian Airlines',
    'VX': 'Virgin America'
}

airport_mapping = {
    'ATL': 'Hartsfield-Jackson Atlanta International Airport',
    'LAX': 'Los Angeles International Airport',
    'ORD': 'Chicago O\'Hare International Airport',
    'DFW': 'Dallas/Fort Worth International Airport',
    'DEN': 'Denver International Airport',
    'JFK': 'John F. Kennedy International Airport',
    'SFO': 'San Francisco International Airport',
    'SEA': 'Seattle-Tacoma International Airport',
    'LAS': 'Harry Reid International Airport',
    'MCO': 'Orlando International Airport'
}

# Use the unique values from your notebook
unique_airlines = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'VX']
unique_airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MCO']

# Main input section
st.header("📋 Flight Details")

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🛫 Origin")
    origin_code = st.selectbox(
        "Select Origin Airport",
        options=unique_airports,
        format_func=lambda x: f"{x} - {airport_mapping.get(x, 'Unknown Airport')}"
    )

with col2:
    st.subheader("🛬 Destination")
    dest_code = st.selectbox(
        "Select Destination Airport",
        options=unique_airports,
        format_func=lambda x: f"{x} - {airport_mapping.get(x, 'Unknown Airport')}"
    )

with col3:
    st.subheader("✈️ Airline")
    airline_code = st.selectbox(
        "Select Airline",
        options=unique_airlines,
        format_func=lambda x: f"{x} - {airline_mapping.get(x, 'Unknown Airline')}"
    )

st.divider()

# Flight schedule
st.header("📅 Flight Schedule")

col4, col5 = st.columns(2)

with col4:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: datetime(2024, x, 1).strftime('%B')
    )
    day = st.selectbox("Day", options=list(range(1, 32)))
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(1, 8)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", 
                              "Thursday", "Friday", "Saturday", "Sunday"][x-1]
    )

with col5:
    scheduled_departure = st.slider("Departure Hour (24-hour)", 0, 23, 12)
    distance = st.slider("Distance (miles)", 50, 3000, 500, 50)
    scheduled_time = st.slider("Flight Time (minutes)", 30, 600, 120, 15)

# Calculate derived features (from your notebook)
hour_of_day = scheduled_departure
is_morning_rush = 1 if hour_of_day in [6, 7, 8] else 0
is_evening_rush = 1 if hour_of_day in [17, 18, 19] else 0
is_night_flight = 1 if hour_of_day in [22, 23, 0, 1, 2, 3, 4, 5] else 0
is_weekend = 1 if day_of_week in [6, 7] else 0
winter_month = 1 if month in [12, 1, 2] else 0
summer_month = 1 if month in [6, 7, 8] else 0
holiday_season = 1 if month in [11, 12] else 0
is_short_flight = 1 if distance < 500 else 0
is_long_flight = 1 if distance > 2000 else 0

# Calculate scheduled arrival (HHMM format)
scheduled_arrival_hhmm = (scheduled_departure * 100 + scheduled_time) % 2400

# Show derived features
st.divider()
st.header("📊 Derived Features")

col6, col7, col8 = st.columns(3)

with col6:
    st.metric("Rush Hour", 
              "Morning" if is_morning_rush else 
              "Evening" if is_evening_rush else "Normal")
    st.metric("Weekend", "Yes" if is_weekend else "No")

with col7:
    season = "Winter" if winter_month else \
             "Summer" if summer_month else \
             "Holiday" if holiday_season else "Regular"
    st.metric("Season", season)
    st.metric("Night Flight", "Yes" if is_night_flight else "No")

with col8:
    flight_length = "Short" if is_short_flight else \
                   "Long" if is_long_flight else "Medium"
    st.metric("Flight Length", flight_length)
    st.metric("Distance", f"{distance} miles")

# Create input DataFrame (EXACTLY as your model expects)
input_data = pd.DataFrame([{
    'ORIGIN_AIRPORT': origin_code,
    'AIRLINE': airline_code,
    'DESTINATION_AIRPORT': dest_code,
    'MONTH': month,
    'DAY': day,
    'DAY_OF_WEEK': day_of_week,
    'SCHEDULED_DEPARTURE': scheduled_departure * 100,  # Convert to HHMM format
    'SCHEDULED_ARRIVAL': scheduled_arrival_hhmm,
    'SCHEDULED_TIME': scheduled_time,
    'DISTANCE': distance,
    'hour_of_day': hour_of_day,
    'is_morning_rush': is_morning_rush,
    'is_evening_rush': is_evening_rush,
    'is_night_flight': is_night_flight,
    'is_weekend': is_weekend,
    'winter_month': winter_month,
    'summer_month': summer_month,
    'holiday_season': holiday_season,
    'is_short_flight': is_short_flight,
    'is_long_flight': is_long_flight
}])

# Prediction button
st.divider()
st.header("🎯 Prediction")

if model is None:
    st.warning("⚠️ Model not loaded. Running in demo mode.")
    
    if st.button("Demo Prediction", type="primary", use_container_width=True):
        # Demo predictions
        demo_delay_prob = np.random.uniform(0.2, 0.8)
        if demo_delay_prob > 0.5:
            st.error(f"⚠️ **Likely DELAYED** ({demo_delay_prob:.1%} probability)")
        else:
            st.success(f"✅ **Likely ON TIME** ({1-demo_delay_prob:.1%} probability)")
        
        st.info("💡 This is a demo. Upload 'model_protocol4.pkl' for real predictions.")
else:
    if st.button("Predict Delay", type="primary", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(input_data)
            
            # Get probabilities if available
            try:
                probabilities = model.predict_proba(input_data)
                delay_prob = probabilities[0][1] * 100
                on_time_prob = probabilities[0][0] * 100
            except:
                # If no predict_proba, use simple prediction
                delay_prob = 100 if prediction[0] == 1 else 0
                on_time_prob = 100 if prediction[0] == 0 else 0
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
                st.progress(delay_prob / 100)
            
            with col_result2:
                if prediction[0] == 1:
                    st.error("⚠️ **FLIGHT LIKELY TO BE DELAYED**")
                    st.markdown(f"**Confidence**: {delay_prob:.1f}%")
                else:
                    st.success("✅ **FLIGHT LIKELY TO BE ON TIME**")
                    st.markdown(f"**Confidence**: {on_time_prob:.1f}%")
            
            # Show flight summary
            with st.expander("📋 View Flight Details"):
                st.write(f"**Route**: {airline_mapping.get(airline_code, airline_code)} from {airport_mapping.get(origin_code, origin_code)} to {airport_mapping.get(dest_code, dest_code)}")
                st.write(f"**Date**: {datetime(2024, month, day).strftime('%B %d')} ({['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][day_of_week-1]})")
                st.write(f"**Departure**: {scheduled_departure:02d}:00")
                st.write(f"**Flight Time**: {scheduled_time} minutes")
                st.write(f"**Distance**: {distance} miles")
                
                # Show risk factors
                st.write("**Risk Factors**:")
                if is_morning_rush or is_evening_rush:
                    st.write("• Rush hour flight")
                if is_weekend:
                    st.write("• Weekend travel")
                if winter_month:
                    st.write("• Winter season")
                if holiday_season:
                    st.write("• Holiday season")
                if is_short_flight:
                    st.write("• Short flight (<500 miles)")
                if is_long_flight:
                    st.write("• Long flight (>2000 miles)")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction[0] == 1:
                st.warning("""
                Consider these options:
                - Book an earlier flight if possible
                - Allow extra time for connections
                - Check flight status before heading to airport
                - Consider travel insurance
                """)
            else:
                st.info("""
                Your flight looks good!
                - Standard arrival time should be fine
                - Still check flight status before departure
                - Have a safe trip!
                """)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            
            # Debug information
            with st.expander("🛠️ Debug Information"):
                st.write("**Model type**:", type(model))
                st.write("**Input columns**:", list(input_data.columns))
                st.write("**Input data shape**:", input_data.shape)
                
                if hasattr(model, 'n_features_in_'):
                    st.write("**Model expects features**:", model.n_features_in_)
                elif hasattr(model, 'feature_names_in_'):
                    st.write("**Model feature names**:", list(model.feature_names_in_))

# Footer
st.divider()
st.caption("""
**Note**: Predictions are based on historical data. Actual delays may vary due to weather, 
air traffic control, or operational factors. Always check with your airline for the most 
up-to-date flight information.
""")

# Requirements for deployment
st.sidebar.divider()
st.sidebar.header("📦 Requirements")
st.sidebar.code("""
streamlit==1.28.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
""")