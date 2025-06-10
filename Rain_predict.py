import streamlit as st
st.set_page_config(
    page_title="Rain Tomorrow Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1601197987860-65d84d8b8f17");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stSidebar {
        background-color: #1c1c1e !important;
    }
    </style>
""", unsafe_allow_html=True)


import pandas as pd
import numpy as np
import math
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.joblib')

# Direction to degrees
DIRECTION_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}

# Location one-hot columns
location_columns = [
    'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat', 'Bendigo',
    'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor',
    'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 'Launceston', 'Melbourne',
    'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 'MountGinini',
    'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
    'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale',
    'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong',
    'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown',
    'Witchcliffe', 'Wollongong', 'Woomera'
]

cols_to_scale = [
    'MinTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud3pm', 'Temp3pm', 'year'
]

# --- UI ---
st.title("🌦️ Rain Tomorrow Prediction")
st.markdown("Predict whether it will rain tomorrow in Australia using weather features and a trained neural network model.")

# --- Inputs ---
with st.sidebar:
    st.header("📋 Input Features")
    date = st.date_input("Select Date", datetime.today())
    selected_location = st.selectbox("Select Location", location_columns)
    min_temp = st.number_input("Min Temperature (°C)", value=10.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.0)
    gust_dir = st.selectbox("Gust Direction", DIRECTION_DEGREES)
    wind9am_dir = st.selectbox("Wind Direction at 9 AM", DIRECTION_DEGREES)
    wind3pm_dir = st.selectbox("Wind Direction at 3 PM", DIRECTION_DEGREES)
    wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", value=30.0)
    wind_speed_3pm = st.number_input("Wind Speed at 3 PM (km/h)", value=20.0)
    humidity_9am = st.slider("Humidity at 9 AM (%)", 0, 100, 50)
    humidity_3pm = st.slider("Humidity at 3 PM (%)", 0, 100, 50)
    pressure_9am = st.number_input("Pressure at 9 AM (hPa)", value=1015.0)
    cloud_3pm = st.slider("Cloud Cover at 3 PM (oktas)", 0, 8, 3)
    temp_3pm = st.number_input("Temperature at 3 PM (°C)", value=20.0)
    rain_today = st.radio("Did it rain today?", ["No", "Yes"])

# --- Helpers ---
def encode_cyclic(value, max_val):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

def encode_wind_dir(dir_str):
    deg = DIRECTION_DEGREES.get(dir_str, 0)
    return math.sin(math.radians(deg)), math.cos(math.radians(deg))

# --- Feature Engineering ---
month_sin, month_cos = encode_cyclic(date.month, 12)
day_sin, day_cos = encode_cyclic(date.day, 31)
rain_today_val = 1 if rain_today == "Yes" else 0
wind_gust_dir_sin, wind_gust_dir_cos = encode_wind_dir(gust_dir)
wind_dir_9am_sin, wind_dir_9am_cos = encode_wind_dir(wind9am_dir)
wind_dir_3pm_sin, wind_dir_3pm_cos = encode_wind_dir(wind3pm_dir)

# --- Input Dictionary ---
input_dict = {
    'MinTemp': min_temp,
    'Rainfall': rainfall,
    'WindGustSpeed': wind_gust_speed,
    'WindSpeed3pm': wind_speed_3pm,
    'Humidity9am': humidity_9am,
    'Humidity3pm': humidity_3pm,
    'Pressure9am': pressure_9am,
    'Cloud3pm': cloud_3pm,
    'Temp3pm': temp_3pm,
    'RainToday': rain_today_val,
    'year': date.year,
    'month_sin': month_sin,
    'month_cos': month_cos,
    'day_sin': day_sin,
    'day_cos': day_cos,
    'WindGustDir_sin': wind_gust_dir_sin,
    'WindGustDir_cos': wind_gust_dir_cos,
    'WindDir9am_sin': wind_dir_9am_sin,
    'WindDir9am_cos': wind_dir_9am_cos,
    'WindDir3pm_sin': wind_dir_3pm_sin,
    'WindDir3pm_cos': wind_dir_3pm_cos,
}

# One-hot encode location
for loc in location_columns:
    input_dict[f"Location_{loc}"] = 1 if loc == selected_location else 0

# --- Prediction ---
input_df = pd.DataFrame([input_dict])
input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
prediction = model.predict(input_df.values)[0][0]
result = "🌧️ Yes" if prediction > 0.5 else "☀️ No"
confidence = prediction if prediction > 0.5 else 1 - prediction

# --- Output ---
st.markdown("## 🌤️ Prediction Result")
st.subheader(f"**Tomorrow's Rain Forecast:** {result}")
st.markdown(f"<div style='font-size: 20px;'>Confidence: <b>{confidence:.2%}</b></div>", unsafe_allow_html=True)


