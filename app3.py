import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="✈️ Flight Delay Predictor", layout="centered")

st.title("✈️ Flight Delay Predictor")
st.markdown("Predict whether your flight will be delayed by 15+ minutes.")

with st.sidebar:
    st.header("Project Information")
    
    st.info("""
    Student ID: PIUS20230024 
    Student Name: May Mon Thant 
    Course: Introduction to Machine Learning  
    University: Parami University  
    Instructor: Prof. Nwe Nwe Htay Win
    """)

@st.cache_resource
def load_model():
    try:
        return joblib.load("flight_delay2.pkl")
    except:
        st.error("Model file 'flight_delay.pkl' not found or failed to load.")
        return None

model = load_model()

airline_mapping = {
    'UA': 'United Air Lines Inc. (UA)',
    'AA': 'American Airlines Inc. (AA)',
    'US': 'US Airways Inc. (US)',
    'F9': 'Frontier Airlines Inc. (F9)',
    'B6': 'JetBlue Airways (B6)',
    'OO': 'Skywest Airlines Inc. (OO)',
    'AS': 'Alaska Airlines Inc. (AS)',
    'NK': 'Spirit Air Lines (NK)',
    'WN': 'Southwest Airlines Co. (WN)',
    'DL': 'Delta Air Lines Inc. (DL)',
    'EV': 'Atlantic Southeast Airlines (EV)',
    'HA': 'Hawaiian Airlines Inc. (HA)',
    'MQ': 'American Eagle Airlines Inc. (MQ)',
    'VX': 'Virgin America (VX)'
}

airport_mapping = {
    'ATL': 'Atlanta Hartsfield-Jackson (ATL)',
    'LAX': 'Los Angeles International (LAX)',
    'ORD': "Chicago O'Hare International (ORD)",
    'DFW': 'Dallas/Fort Worth International (DFW)',
    'DEN': 'Denver International (DEN)',
    'JFK': 'New York JFK International (JFK)',
    'SFO': 'San Francisco International (SFO)',
    'SEA': 'Seattle-Tacoma International (SEA)',
    'LAS': 'Las Vegas McCarran International (LAS)',
    'MCO': 'Orlando International (MCO)',
}

st.header("Flight Information")

origin = st.selectbox(
    "Origin Airport",
    list(airport_mapping.keys()),
    format_func=lambda x: airport_mapping[x]
)

destination = st.selectbox(
    "Destination Airport",
    list(airport_mapping.keys()),
    format_func=lambda x: airport_mapping[x]
)

airline = st.selectbox(
    "Airline",
    list(airline_mapping.keys()),
    format_func=lambda x: airline_mapping[x]
)

month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=1)
day_of_week = st.selectbox(
    "Day of Week",
    [1,2,3,4,5,6,7],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1]
)

dep_hour = st.number_input("Departure Hour (0–23)", min_value=0, max_value=23, value=12)
duration = st.number_input("Scheduled Duration (minutes)", min_value=20, max_value=600, value=120)
distance = st.number_input("Distance (miles)", min_value=50, max_value=3000, value=500)

arrival_time = (dep_hour * 100 + duration) % 2400

input_df = pd.DataFrame([{
    "ORIGIN_AIRPORT": origin,
    "DESTINATION_AIRPORT": destination,
    "AIRLINE": airline,
    "MONTH": month,
    "DAY": day,
    "DAY_OF_WEEK": day_of_week,
    "SCHEDULED_DEPARTURE": dep_hour * 100,
    "SCHEDULED_ARRIVAL": arrival_time,
    "SCHEDULED_TIME": duration,
    "DISTANCE": distance
}])

st.write("### Model Input Data")
st.dataframe(input_df)


if st.button("Predict Delay"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] * 100

            st.subheader("Prediction Result")

            if pred == 1:
                st.error(f"**Flight Likely Delayed ({prob:.1f}% chance)**")
            else:
                st.success(f" **Flight Likely On Time ({100-prob:.1f}% chance)**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
