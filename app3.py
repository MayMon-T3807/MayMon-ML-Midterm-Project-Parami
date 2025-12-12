import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flight Delay Predictor", layout="centered")

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
        return joblib.load("flight_model.pkl")
    except:
        st.error("Model file 'flight_model.pkl' not found or failed to load.")
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
    'ATL': 'Atlanta Hartsfield-Jackson (ATL)', 'LAX': 'Los Angeles International (LAX)',
    'ORD': "Chicago O'Hare International (ORD)", 'DFW': 'Dallas/Fort Worth International (DFW)',
    'DEN': 'Denver International (DEN)', 'JFK': 'New York JFK International (JFK)',
    'SFO': 'San Francisco International (SFO)', 'SEA': 'Seattle-Tacoma International (SEA)',
    'LAS': 'Las Vegas McCarran International (LAS)', 'MCO': 'Orlando International (MCO)',
    'CLT': 'Charlotte Douglas International (CLT)', 'MIA': 'Miami International (MIA)',
    'PHX': 'Phoenix Sky Harbor International (PHX)', 'IAH': 'Houston George Bush Intercontinental (IAH)',
    'BOS': 'Boston Logan International (BOS)', 'MSP': 'Minneapolis-Saint Paul International (MSP)',
    'FLL': 'Fort Lauderdale-Hollywood International (FLL)', 'DTW': 'Detroit Metropolitan (DTW)',
    'PHL': 'Philadelphia International (PHL)', 'LGA': 'New York LaGuardia (LGA)',
    'BWI': 'Baltimore/Washington International (BWI)', 'SLC': 'Salt Lake City International (SLC)',
    'SAN': 'San Diego International (SAN)', 'IAD': 'Washington Dulles International (IAD)',
    'DCA': 'Washington Reagan National (DCA)', 'MDW': 'Chicago Midway International (MDW)',
    'TPA': 'Tampa International (TPA)', 'PDX': 'Portland International (PDX)',
    'HNL': 'Honolulu International (HNL)', 'STL': 'St. Louis Lambert International (STL)',
    'BNA': 'Nashville International (BNA)', 'AUS': 'Austin-Bergstrom International (AUS)',
    'MSY': 'New Orleans Louis Armstrong International (MSY)', 'RDU': 'Raleigh-Durham International (RDU)',
    'MCI': 'Kansas City International (MCI)', 'SJC': 'San Jose International (SJC)',
    'SMF': 'Sacramento International (SMF)', 'SAT': 'San Antonio International (SAT)',
    'CVG': 'Cincinnati/Northern Kentucky International (CVG)', 'CLE': 'Cleveland Hopkins International (CLE)',
    'IND': 'Indianapolis International (IND)', 'CMH': 'Columbus International (CMH)',
    'PIT': 'Pittsburgh International (PIT)', 'MKE': 'Milwaukee Mitchell International (MKE)',
    'OMA': 'Omaha Eppley Airfield (OMA)', 'BUF': 'Buffalo Niagara International (BUF)',
    'MEM': 'Memphis International (MEM)', 'ABQ': 'Albuquerque International Sunport (ABQ)',
    'TUS': 'Tucson International (TUS)', 'OKC': 'Oklahoma City Will Rogers World (OKC)',
    'TUL': 'Tulsa International (TUL)', 'ANC': 'Anchorage Ted Stevens International (ANC)',
    'FAI': 'Fairbanks International (FAI)', 'ELP': 'El Paso International (ELP)',
    'ALB': 'Albany International (ALB)', 'BHM': 'Birmingham-Shuttlesworth International (BHM)',
    'DAY': 'Dayton International (DAY)', 'GSO': 'Greensboro Piedmont Triad International (GSO)',
    'GRR': 'Grand Rapids Gerald R. Ford International (GRR)', 'HSV': 'Huntsville International (HSV)',
    'JAX': 'Jacksonville International (JAX)', 'LIT': 'Little Rock National (LIT)',
    'PBI': 'West Palm Beach International (PBI)', 'RNO': 'Reno/Tahoe International (RNO)',
    'ROC': 'Rochester International (ROC)', 'SDF': 'Louisville International (SDF)',
    'SYR': 'Syracuse Hancock International (SYR)', 'TYS': 'Knoxville McGhee Tyson (TYS)'
}

st.header("Flight Information")

origin = st.selectbox("Origin Airport", list(airport_mapping.keys()), format_func=lambda x: airport_mapping[x])
destination = st.selectbox("Destination Airport", list(airport_mapping.keys()), format_func=lambda x: airport_mapping[x])
airline = st.selectbox("Airline", list(airline_mapping.keys()), format_func=lambda x: airline_mapping[x])

month = st.number_input("Month (1-12)", 1, 12, 1)
day = st.number_input("Day (1-31)", 1, 31, 1)
day_of_week = st.selectbox("Day of Week", [1,2,3,4,5,6,7])

dep_hour = st.number_input("Departure Hour (0–23)", 0, 23, 12)
duration = st.number_input("Scheduled Duration (minutes)", 20, 600, 120)
distance = st.number_input("Distance (miles)", 50, 3000, 500)

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

def validate(df):
    if df["ORIGIN_AIRPORT"].iloc[0] == df["DESTINATION_AIRPORT"].iloc[0]:
        st.error("Origin and Destination cannot be the same.")
        return False
    return True

if st.button("Predict Delay"):
    if model is None:
        st.error("Model not loaded.")
    else:
        if not validate(input_df):
            st.stop()
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] * 100
            if pred == 1:
                st.error(f"Flight Likely Delayed ({prob:.1f}% chance)")
            else:
                st.success(f"Flight Likely On Time ({100 - prob:.1f}% chance)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
 
