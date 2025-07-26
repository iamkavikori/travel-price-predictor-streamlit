import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 🔹 Load a small sample dataset from GitHub (or local small_dataset.csv)
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv").head(300)
    df.to_csv("small_dataset.csv", index=False)  
    return df

# 🔹 Train model dynamically
@st.cache_resource
def train_model(df):
    df = df.copy()
    df.dropna(inplace=True)
    
    # Example categorical columns (adjust according to dataset)
    categorical_cols = ['Airline', 'Source', 'Destination','Stop']
    encoders = {}
    
    for col in categorical_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc
    
    # Features and target
    X = df[['Airline','Source','Destination','Duration','Stops']]
    y = df['Price']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, encoders

# 🔹 Load data & train model
df = load_data()
model, encoders = train_model(df)

# 🔹 Unique options for dropdowns
airlines = df['Airline'].unique().tolist()
sources = df['Source'].unique().tolist()
destinations = df['Destination'].unique().tolist()

# 🔹 Streamlit UI
st.title("✈️ Travel Price Predictor")
st.markdown("Predict flight prices based on your input parameters.")

# Dropdown Inputs
airline = st.selectbox("Select Airline", airlines)
source = st.selectbox("Select Source", sources)
destination = st.selectbox("Select Destination", destinations)
duration = st.slider("Duration (minutes)", 30, 1500, 180)
stops = st.slider("Stops", 0, 1)

# Predict Button
if st.button("Predict Price"):
    # Encode inputs
    a = encoders['Airline'].transform([airline])[0]
    s = encoders['Source'].transform([source])[0]
    d = encoders['Destination'].transform([destination])[0]
    
    sample = np.array([[a, s, d, duration, stops]])
    price = model.predict(sample)[0]
    st.success(f"💰 Predicted Flight Price: ₹ {round(price,2)}")
