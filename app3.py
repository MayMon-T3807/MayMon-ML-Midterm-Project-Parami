# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
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
    model_files_to_check = ['flight_delay.pkl', 'flight_delay2.pkl', 'full_pipeline.joblib']
    found_model = None
    
    for model_file in model_files_to_check:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            st.success(f"✅ Found: {model_file} ({file_size:.1f} MB)")
            found_model = model_file
            break
    
    if not found_model:
        st.error("❌ No model file found!")
        st.info("Please upload one of these files:")
        for file in model_files_to_check:
            st.code(file)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model - try multiple methods"""
    model_files = ['flight_delay.pkl', 'flight_delay2.pkl', 'full_pipeline.joblib']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                # Try joblib first
                model = joblib.load(model_file)
                st.sidebar.success(f"✅ Loaded: {model_file} (joblib)")
                return model
            except Exception as e1:
                try:
                    # Try pickle as fallback
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    st.sidebar.success(f"✅ Loaded: {model_file} (pickle)")
                    return model
                except Exception as e2:
                    st.sidebar.warning(f"Failed {model_file}: joblib={str(e1)[:50]}, pickle={str(e2)[:50]}")
                    continue
    
    st.sidebar.error("❌ Could not load any model file")
    return None

# Load the model
model = load_model()

# Default airline and airport data
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
    'ATL': 'Atlanta (ATL)',
    'LAX': 'Los Angeles (LAX)',
    'ORD': 'Chicago O\'Hare (ORD)',
    'DFW': 'Dallas/Fort Worth (DFW)',
    'DEN': 'Denver (DEN)',
    'JFK': 'New York JFK (JFK)',
    'SFO': 'San Francisco (SFO)',
    'SEA': 'Seattle (SEA)',
    'LAS': 'Las Vegas (LAS)',
    'MCO': 'Orlando (MCO)'
}

# Use the unique values
unique_airlines = list(airline_mapping.keys())
unique_airports = list(airport_mapping.keys())

# Main input section
st.header("📋 Flight Details")

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🛫 Origin")
    origin_code = st.selectbox(
        "Select Origin Airport",
        options=unique_airports,
        format_func=lambda x: airport_mapping[x],
        key="origin"
    )

with col2:
    st.subheader("🛬 Destination")
    dest_code = st.selectbox(
        "Select Destination Airport",
        options=unique_airports,
        format_func=lambda x: airport_mapping[x],
        key="dest"
    )

with col3:
    st.subheader("✈️ Airline")
    airline_code = st.selectbox(
        "Select Airline",
        options=unique_airlines,
        format_func=lambda x: airline_mapping[x],
        key="airline"
    )

st.divider()

# Flight schedule
st.header("📅 Flight Schedule")

col4, col5 = st.columns(2)

with col4:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
        key="month"
    )
    day = st.selectbox("Day", options=list(range(1, 32)), key="day")
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(1, 8)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", 
                              "Thursday", "Friday", "Saturday", "Sunday"][x-1],
        key="dow"
    )

with col5:
    scheduled_departure = st.slider("Departure Hour (24-hour)", 0, 23, 12, key="hour")
    distance = st.slider("Distance (miles)", 50, 3000, 500, 50, key="distance")
    scheduled_time = st.slider("Flight Time (minutes)", 30, 600, 120, 15, key="duration")

# Calculate derived features
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

# Show input data for debugging
with st.expander("🔍 View Input Data"):
    st.write("This is what will be sent to the model:")
    st.dataframe(input_data)

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
        
        st.info("💡 Upload 'flight_delay.pkl' for real predictions.")
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
                st.write(f"**Airline**: {airline_mapping[airline_code]}")
                st.write(f"**Route**: {airport_mapping[origin_code]} → {airport_mapping[dest_code]}")
                st.write(f"**Date**: {datetime(2024, month, day).strftime('%B %d')} ({['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][day_of_week-1]})")
                st.write(f"**Departure**: {scheduled_departure:02d}:00")
                st.write(f"**Flight Time**: {scheduled_time} minutes")
                st.write(f"**Distance**: {distance} miles")
                
                # Show risk factors
                risk_factors = []
                if is_morning_rush or is_evening_rush:
                    risk_factors.append("Rush hour flight")
                if is_weekend:
                    risk_factors.append("Weekend travel")
                if winter_month:
                    risk_factors.append("Winter season")
                if holiday_season:
                    risk_factors.append("Holiday season")
                if is_short_flight:
                    risk_factors.append("Short flight (<500 miles)")
                if is_long_flight:
                    risk_factors.append("Long flight (>2000 miles)")
                
                if risk_factors:
                    st.write("**Risk Factors**:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("**Risk Factors**: None (low risk profile)")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction[0] == 1:
                st.warning("""
                **Consider these options:**
                - Book an earlier flight if possible
                - Allow extra time for connections
                - Check flight status before heading to airport
                - Consider travel insurance
                """)
            else:
                st.info("""
                **Your flight looks good!**
                - Standard arrival time should be fine
                - Still check flight status before departure
                - Have a safe trip!
                """)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            
            # Debug information
            with st.expander("🛠️ Debug Information"):
                st.write("**Error type**:", type(e).__name__)
                st.write("**Model type**:", type(model))
                
                if hasattr(model, 'steps'):
                    st.write("**Pipeline steps**:", [step[0] for step in model.steps])
                
                st.write("**Input columns**:", list(input_data.columns))
                st.write("**Input data shape**:", input_data.shape)
                
                # Check if it's a pipeline issue
                if "could not convert string to float" in str(e):
                    st.info("""
                    **Common issue**: The model expects preprocessed data but got raw strings.
                    
                    **Solution**: Make sure you're using the FULL pipeline (flight_delay.pkl),
                    not just the classifier (model_protocol4.pkl).
                    """)
                elif "No module named 'imblearn'" in str(e):
                    st.info("""
                    **imblearn issue**: The model was trained with SMOTE from imblearn.
                    
                    **Solution**: 
                    1. Install imblearn: `pip install imbalanced-learn`
                    2. Or recreate model without imblearn dependencies
                    """)

# Footer
st.divider()
st.caption("""
**Note**: Predictions are based on historical data. Actual delays may vary due to weather, 
air traffic control, or operational factors. Always check with your airline for official flight status.
""")

# Requirements
with st.sidebar:
    st.divider()
    st.header("📦 Requirements")
    st.code("""
streamlit==1.28.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
imbalanced-learn==0.11.0  # Only if using flight_delay.pkl with SMOTE
""")