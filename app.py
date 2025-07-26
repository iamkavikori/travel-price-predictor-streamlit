import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("✈️ Travel Price Predictor")
st.markdown("Predict flight prices based on airline, source, destination, duration & stops.")

# User Inputs
airline = st.number_input("Airline (numeric code)", min_value=0, max_value=10, value=1)
source = st.number_input("Source (numeric code)", min_value=0, max_value=10, value=2)
destination = st.number_input("Destination (numeric code)", min_value=0, max_value=10, value=5)
duration = st.number_input("Duration (minutes)", min_value=30, max_value=1500, value=180)
stops = st.number_input("Total Stops", min_value=0, max_value=5, value=1)

# Prediction
if st.button("Predict Price"):
    sample = np.array([[airline, source, destination, duration, stops]])
    price = model.predict(sample)[0]
    st.success(f"💰 Predicted Flight Price: ₹ {round(price,2)}")