import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("Flight Delay Predictor")
st.markdown("Predict if your flight will be delayed by more than 15 minutes")

with st.sidebar:
    st.header("Student Information")
    st.info("""
    Student ID: PIUS20230024
    Student Name: May Mon Thant
    Course: Introduction to Machine Learning  
    University: Parami University  
    Instructor: Prof. Nwe Nwe Htay Win
    """)

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        try:
            model = joblib.load('flight_model.pkl')
            return model
        except Exception as e1:
            try:
                with open('model.pkl', 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e2:
                st.error(f"Failed to load model: {str(e2)[:100]}")
                return None
    else:
        st.error("Model file 'model.pkl' not found")
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
    'ORD': 'Chicago O\'Hare International (ORD)',
    'DFW': 'Dallas/Fort Worth International (DFW)',
    'DEN': 'Denver International (DEN)',
    'JFK': 'New York JFK International (JFK)',
    'SFO': 'San Francisco International (SFO)',
    'SEA': 'Seattle-Tacoma International (SEA)',
    'LAS': 'Las Vegas McCarran International (LAS)',
    'MCO': 'Orlando International (MCO)',
    'CLT': 'Charlotte Douglas International (CLT)',
    'MIA': 'Miami International (MIA)',
    'PHX': 'Phoenix Sky Harbor International (PHX)',
    'IAH': 'Houston George Bush Intercontinental (IAH)',
    'BOS': 'Boston Logan International (BOS)',
    'MSP': 'Minneapolis-Saint Paul International (MSP)',
    'FLL': 'Fort Lauderdale-Hollywood International (FLL)',
    'DTW': 'Detroit Metropolitan (DTW)',
    'PHL': 'Philadelphia International (PHL)',
    'LGA': 'New York LaGuardia (LGA)',
    'BWI': 'Baltimore/Washington International (BWI)',
    'SLC': 'Salt Lake City International (SLC)',
    'SAN': 'San Diego International (SAN)',
    'IAD': 'Washington Dulles International (IAD)',
    'DCA': 'Washington Reagan National (DCA)',
    'MDW': 'Chicago Midway International (MDW)',
    'TPA': 'Tampa International (TPA)',
    'PDX': 'Portland International (PDX)',
    'HNL': 'Honolulu International (HNL)',
    'STL': 'St. Louis Lambert International (STL)',
    'BNA': 'Nashville International (BNA)',
    'AUS': 'Austin-Bergstrom International (AUS)',
    'MSY': 'New Orleans Louis Armstrong International (MSY)',
    'RDU': 'Raleigh-Durham International (RDU)',
    'MCI': 'Kansas City International (MCI)',
    'SJC': 'San Jose International (SJC)',
    'SMF': 'Sacramento International (SMF)',
    'SAT': 'San Antonio International (SAT)',
    'CVG': 'Cincinnati/Northern Kentucky International (CVG)',
    'CLE': 'Cleveland Hopkins International (CLE)',
    'IND': 'Indianapolis International (IND)',
    'CMH': 'Columbus International (CMH)',
    'PIT': 'Pittsburgh International (PIT)',
    'MKE': 'Milwaukee Mitchell International (MKE)',
    'OMA': 'Omaha Eppley Airfield (OMA)',
    'BUF': 'Buffalo Niagara International (BUF)',
    'MEM': 'Memphis International (MEM)',
    'ABQ': 'Albuquerque International Sunport (ABQ)',
    'TUS': 'Tucson International (TUS)',
    'OKC': 'Oklahoma City Will Rogers World (OKC)',
    'TUL': 'Tulsa International (TUL)',
    'ANC': 'Anchorage Ted Stevens International (ANC)',
    'FAI': 'Fairbanks International (FAI)',
    'ELP': 'El Paso International (ELP)',
    'ALB': 'Albany International (ALB)',
    'BHM': 'Birmingham-Shuttlesworth International (BHM)',
    'DAY': 'Dayton International (DAY)',
    'GSO': 'Greensboro Piedmont Triad International (GSO)',
    'GRR': 'Grand Rapids Gerald R. Ford International (GRR)',
    'HSV': 'Huntsville International (HSV)',
    'JAX': 'Jacksonville International (JAX)',
    'LIT': 'Little Rock Bill and Hillary Clinton National (LIT)',
    'PBI': 'West Palm Beach International (PBI)',
    'RNO': 'Reno/Tahoe International (RNO)',
    'ROC': 'Rochester International (ROC)',
    'SDF': 'Louisville International (SDF)',
    'SYR': 'Syracuse Hancock International (SYR)',
    'TYS': 'Knoxville McGhee Tyson (TYS)',
}

unique_airlines = list(airline_mapping.keys())
unique_airports = list(airport_mapping.keys())

st.header("Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Origin")
    origin_code = st.selectbox(
        "Select Origin Airport",
        options=unique_airports,
        format_func=lambda x: airport_mapping[x],
        key="origin"
    )

with col2:
    st.subheader("Destination")
    dest_code = st.selectbox(
        "Select Destination Airport",
        options=unique_airports,
        format_func=lambda x: airport_mapping[x],
        key="dest"
    )

with col3:
    st.subheader("Airline")
    airline_code = st.selectbox(
        "Select Airline",
        options=unique_airlines,
        format_func=lambda x: airline_mapping[x],
        key="airline"
    )

st.divider()

st.header("Flight Schedule")

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
    departure_hours = [f"{hour:02d}:00" for hour in range(0, 24)]
    departure_hour_display = st.selectbox(
        "Departure Hour (24-hour)",
        options=departure_hours,
        index=12, 
        key="hour_display"
    )
    scheduled_departure = int(departure_hour_display.split(":")[0])  
    
    distance = st.slider("Distance (miles)", 50, 3000, 500, 50, key="distance")
    
    flight_times = list(range(30, 601, 15))  
    flight_time_options = [f"{time // 60}h {time % 60}min" if time >= 60 else f"{time}min" for time in flight_times]
    flight_time_display = st.selectbox(
        "Flight Time",
        options=flight_time_options,
        index=flight_times.index(120),  
        key="duration_display"
    )
    scheduled_time = flight_times[flight_time_options.index(flight_time_display)]

st.divider()


scheduled_arrival_hhmm = (scheduled_departure * 100 + scheduled_time) % 2400

input_data = pd.DataFrame([{
    'ORIGIN_AIRPORT': origin_code,
    'AIRLINE': airline_code,
    'DESTINATION_AIRPORT': dest_code,
    'MONTH': month,
    'DAY': day,
    'DAY_OF_WEEK': day_of_week,
    'SCHEDULED_DEPARTURE': scheduled_departure * 100,
    'SCHEDULED_ARRIVAL': scheduled_arrival_hhmm,
    'SCHEDULED_TIME': scheduled_time,
    'DISTANCE': distance
}])

with st.expander("View Input Data"):
    st.write("This is what will be sent to the model:")
    st.dataframe(input_data)

st.divider()
st.header("Prediction")

if st.button("Predict Delay", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded. Please ensure 'flight_model.pkl' exists in the current directory.")
    else:
        try:
            prediction = model.predict(input_data)
            
            try:
                probabilities = model.predict_proba(input_data)
                delay_prob = probabilities[0][1] * 100
                on_time_prob = probabilities[0][0] * 100
            except:
                delay_prob = 100 if prediction[0] == 1 else 0
                on_time_prob = 100 if prediction[0] == 0 else 0
            
            st.subheader("Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
                st.progress(delay_prob / 100)
            
            with col_result2:
                if prediction[0] == 1:
                    st.error("FLIGHT LIKELY TO BE DELAYED")
                else:
                    st.success("FLIGHT LIKELY TO BE ON TIME")
            
            with st.expander("View Flight Details"):
                st.write(f"Airline: {airline_mapping[airline_code]}")
                st.write(f"Route: {airport_mapping[origin_code]} to {airport_mapping[dest_code]}")
                st.write(f"Date: {datetime(2024, month, day).strftime('%B %d')} ({['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][day_of_week-1]})")
                st.write(f"Departure: {departure_hour_display}")
                st.write(f"Flight Time: {flight_time_display}")
                st.write(f"Distance: {distance} miles")
                
                if distance < 500:
                    flight_type = "Short (<500 miles)"
                elif distance <= 2000:
                    flight_type = "Medium (500-2000 miles)"
                else:
                    flight_type = "Long (>2000 miles)"
                st.write(f"Flight Type: {flight_type}")
            
            st.subheader("Recommendations")
            if prediction[0] == 1:
                st.warning("""
                Book an earlier flight if possible
                """)
            else:
                st.info("""
                No sign of delay and Have a safe trip!
                """)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            
            with st.expander("Debug Information"):
                st.write("Error type:", type(e).__name__)
                st.write("Model type:", type(model))
                
                if hasattr(model, 'steps'):
                    st.write("Pipeline steps:", [step[0] for step in model.steps])
                
                st.write("Input columns:", list(input_data.columns))
                st.write("Input data shape:", input_data.shape)

st.divider()
st.caption("""
Note: Predictions are based on historical data. Actual delays may vary due to weather,
air traffic control, or operational factors. Always check with your airline for official flight status.
""")