import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
import sklearn

print(sklearn.__version__)


st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("Flight Delay Predictor")
st.markdown("Predict whether your flight will be delayed by more than **15 minutes**.")


with st.sidebar:
    st.header("Student Information")
    st.info("""
    **Student ID:** PIUS20230024  
    **Student Name:** May Mon Thant  
    **Course:** Introduction to Machine Learning  
    **University:** Parami University  
    **Instructor:** Prof. Nwe Nwe Htay Win
    """)


@st.cache_resource
def load_model():
    model_path = "flight_model2.pkl"   

    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found in directory.")
        return None

    try:
        return joblib.load(model_path)
    except:
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"ERROR loading model: {str(e)}")
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
    'MCO': 'Orlando International (MCO)'
}

unique_airlines = list(airline_mapping.keys())
unique_airports = list(airport_mapping.keys())


st.header("Flight Details")

col1, col2, col3 = st.columns(3)

with col1:
    origin_code = st.selectbox(
        "Origin Airport",
        unique_airports,
        format_func=lambda x: airport_mapping[x],
    )

with col2:
    dest_code = st.selectbox(
        "Destination Airport",
        unique_airports,
        format_func=lambda x: airport_mapping[x],
    )

with col3:
    airline_code = st.selectbox(
        "Airline",
        unique_airlines,
        format_func=lambda x: airline_mapping[x],
    )

st.divider()


st.header("Flight Schedule")

col4, col5 = st.columns(2)

with col4:
    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda x: datetime(2024, x, 1).strftime('%B')
    )

    day = st.selectbox("Day", list(range(1, 32)))

    day_of_week = st.selectbox(
        "Day of Week",
        list(range(1, 8)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x-1]
    )

with col5:
    departure_hours = [f"{hour:02d}:00" for hour in range(24)]
    departure_hour_display = st.selectbox(
        "Departure Hour",
        departure_hours,
        index=12
    )

    scheduled_departure = int(departure_hour_display.split(":")[0]) * 100

    distance = st.slider("Distance (miles)", 50, 3000, 500, 50)

    flight_times = list(range(30, 601, 15))
    flight_time_display = st.selectbox(
        "Flight Duration",
        [f"{t//60}h {t%60}m" for t in flight_times],
        index=flight_times.index(120)
    )
    scheduled_time = flight_times[[f"{t//60}h {t%60}m" for t in flight_times].index(flight_time_display)]


arrival_hour = (scheduled_departure // 100) + (scheduled_time // 60)
arrival_minute = scheduled_time % 60
scheduled_arrival = (arrival_hour % 24) * 100 + arrival_minute


input_data = pd.DataFrame([{
    'ORIGIN_AIRPORT': origin_code,
    'AIRLINE': airline_code,
    'DESTINATION_AIRPORT': dest_code,
    'MONTH': month,
    'DAY': day,
    'DAY_OF_WEEK': day_of_week,
    'SCHEDULED_DEPARTURE': scheduled_departure,
    'SCHEDULED_ARRIVAL': scheduled_arrival,
    'SCHEDULED_TIME': scheduled_time,
    'DISTANCE': distance
}])

with st.expander("View Input Data Sent to Model"):
    st.dataframe(input_data)


st.divider()
st.header("Prediction")

if st.button("Predict Delay", use_container_width=True):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            prediction = model.predict(input_data)[0]

            # Probability
            try:
                proba = model.predict_proba(input_data)[0]
                on_time_prob = proba[0] * 100
                delay_prob = proba[1] * 100
            except:
                delay_prob = 100 if prediction == 1 else 0
                on_time_prob = 100 - delay_prob

            st.subheader("Prediction Results")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
                st.progress(delay_prob / 100)

            with colB:
                if prediction == 1:
                    st.error("⚠️ FLIGHT LIKELY TO BE DELAYED")
                else:
                    st.success("✅ FLIGHT LIKELY TO BE ON TIME")

        except Exception as e:
            st.error(f"Prediction ERROR: {str(e)}")

st.divider()
st.caption("Predictions are based on historical data. Always check official flight status.")
