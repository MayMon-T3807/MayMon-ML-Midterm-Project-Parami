# MayMon-ML-Midterm-Project-Parami

Predict whether your flight will be delayed by more than 15 minutes using machine learning.
This project helps passengers know if their flight might be delayed before they get to the airport. It uses historical flight data to make predictions based on airline, airports, date, time, and distance.
To create a smart system that predicts flight delays so passengers can plan better and avoid wasting time at the airport.

Collect Data: Load flight information (airline, airports, time, distance)
Clean Data: Remove cancelled and diverted flights
Create Features: Add helpful features like rush hour, weekend, season
Handle Imbalance: Use SMOTE to balance delayed vs on-time flights
Train Model: Use Random Forest with 200 decision trees
Make Predictions: Predict if a flight will be delayed

├── flights.csv                 # Flight data
├── airports.csv                # Airport information
├── airlines.csv                # Airline information
├── flight_delay_MayMon.ipynb # Main analysis notebook
├── flight_delay2.pkl      # Trained model
├── app3.py                      # Streamlit web app
└── README.md                   # Project documentation
