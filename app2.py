# app_simple_fixed.py
import streamlit as st
import pandas as pd
import joblib
import pickle

# Simple model loading - just try protocol4.pkl
try:
    model = joblib.load('model_protocol4.pkl')
    st.success("✅ Model loaded successfully with joblib")
except:
    try:
        # Fallback to pickle
        with open('model_protocol4.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully with pickle")
    except:
        st.error("❌ Could not load model_protocol4.pkl")
        st.stop()

# App configuration
st.set_page_config(page_title="Flight Delay Predictor", layout="centered")
st.title("✈️ Flight Delay Predictor")
st.markdown("Predict if your flight will be delayed > 15 minutes")

# Simple input section
col1, col2 = st.columns(2)

with col1:
    # Hardcoded options for simplicity
    airline = st.selectbox("Airline", ["AA", "DL", "UA", "WN", "B6"])
    origin = st.selectbox("Origin", ["ATL", "LAX", "ORD", "DFW", "JFK"])
    destination = st.selectbox("Destination", ["LAX", "ATL", "ORD", "DFW", "JFK"])
    month = st.selectbox("Month", list(range(1, 13)))

with col2:
    day = st.selectbox("Day", list(range(1, 32)))
    day_of_week = st.selectbox("Day of Week", 
                              options=range(1, 8),
                              format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x-1])
    hour = st.slider("Departure Hour", 0, 23, 12)
    duration = st.slider("Flight Time (min)", 60, 600, 180)
    distance = st.slider("Distance (miles)", 100, 3000, 1000)

# Calculate derived features
is_morning_rush = 1 if hour in [6, 7, 8] else 0
is_evening_rush = 1 if hour in [17, 18, 19] else 0
is_night_flight = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
is_weekend = 1 if day_of_week in [6, 7] else 0
winter_month = 1 if month in [12, 1, 2] else 0
summer_month = 1 if month in [6, 7, 8] else 0
holiday_season = 1 if month in [11, 12] else 0
is_short_flight = 1 if distance < 500 else 0
is_long_flight = 1 if distance > 2000 else 0

# Create input dataframe
input_data = pd.DataFrame([{
    'ORIGIN_AIRPORT': origin,
    'AIRLINE': airline,
    'DESTINATION_AIRPORT': destination,
    'MONTH': month,
    'DAY': day,
    'DAY_OF_WEEK': day_of_week,
    'SCHEDULED_DEPARTURE': hour * 100,  # Convert to HHMM
    'SCHEDULED_ARRIVAL': (hour * 100 + duration) % 2400,
    'SCHEDULED_TIME': duration,
    'DISTANCE': distance,
    'hour_of_day': hour,
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
if st.button("Predict Delay", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(input_data)
        
        # Try to get probabilities if available
        try:
            probabilities = model.predict_proba(input_data)
            delay_prob = probabilities[0][1] * 100
            on_time_prob = probabilities[0][0] * 100
        except:
            # If no predict_proba, use simple prediction
            delay_prob = 100 if prediction[0] == 1 else 0
            on_time_prob = 100 if prediction[0] == 0 else 0
        
        # Show results
        st.subheader("🎯 Prediction Result")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **Likely DELAYED** ({delay_prob:.1f}% probability)")
        else:
            st.success(f"✅ **Likely ON TIME** ({on_time_prob:.1f}% probability)")
        
        # Show progress bar
        st.progress(delay_prob / 100)
        
        # Show summary
        st.divider()
        st.subheader("📊 Flight Summary")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Route**: {origin} → {destination}")
            st.write(f"**Airline**: {airline}")
            st.write(f"**Date**: Month {month}, Day {day}")
        
        with col_b:
            st.write(f"**Departure**: {hour:02d}:00")
            st.write(f"**Duration**: {duration} min")
            st.write(f"**Distance**: {distance} miles")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        
        # Debug info
        with st.expander("Debug Info"):
            st.write("Model type:", type(model))
            st.write("Input data columns:", list(input_data.columns))
            st.write("Input data shape:", input_data.shape)
            
            # Show sample of what model expects
            if hasattr(model, 'feature_names_in_'):
                st.write("Model expects:", list(model.feature_names_in_))

# Footer
st.caption("Note: Predictions are based on historical flight data.")