
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import requests
import os
from collections import deque
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
print("=" * 50)
print("PYTHON:", sys.executable)
print("TF VERSION:", tf.__version__)
print("=" * 50)
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API KEY
# =========================
API_KEY = "9e37120cfdb54781b8371238261404"

# =========================
# LOAD MODELS
# =========================
# Updated to v4
rain_model = joblib.load("training/rain_modelv4.pkl")
storm_model = joblib.load("training/thunderstorm_xgb_model.pkl")
heat_model = joblib.load("training/heat_model.pkl")
pollution_model = joblib.load("training/pollution_model.pkl")
aqi_model = joblib.load("training/aqi_model.pkl")
lstm_model = load_model(
    "training/weather_lstm_model.h5",
    compile=False,
    safe_mode=False
)
lstm_scaler = joblib.load(
    "training/lstm_scaler.pkl"
)
lstm_config = joblib.load(
    "training/lstm_config.pkl"
)

rain_threshold = 0.65

# =========================
# FEATURE POOL
# =========================
base_features = [
    'lat','lon','temperature_C','humidity_pct','pressure_hPa',
    'dew_point_C','pressure_trend','solar_radiation_Wm2',
    'wind_speed_ms','cloud_cover_pct','hour','month',
    'wind_direction_deg','wind_dir_sin','wind_dir_cos',
    'et0_mm','precip_mm','city_encoded',
    'temp_dew_diff','humidity_pressure'
]
lstm_fallback_features = [
    'temperature_C',
    'humidity_pct',
    'pressure_hPa',
    'dew_point_C',
    'pressure_trend',
    'solar_radiation_Wm2',
    'wind_speed_ms',
    'cloud_cover_pct',
    'hour',
    'month',
    'wind_direction_deg',
    'et0_mm',
    'precip_mm',
    'wind_dir_sin',
    'wind_dir_cos',
    'lat',
    'lon',
    'city_encoded'
]

# =========================
# WIND DIRECTION TEXT
# =========================
def get_wind_direction(deg):
    directions = ["N","NE","E","SE","S","SW","W","NW"]
    return directions[round(deg / 45) % 8]

# =========================
# WEATHER FETCH
# =========================
def get_weather(query: str):
    query = query.strip()
    if "," not in query:
        query = f"{query}, India"
    
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={query}&aqi=yes"
    res = requests.get(url, timeout=10)
    data = res.json()

    if "current" not in data:
        raise Exception(data.get("error", {}).get("message", "Weather API failed"))

    current = data["current"]
    location = data["location"]
    air = current.get("air_quality", {})
    if not air:
        air = {
            "co": 0, "no2": 0, "o3": 0, "so2": 0, "pm2_5": 0, "pm10": 0
        }

    wind_deg = current["wind_degree"]
    hour = int(location["localtime"].split(" ")[1].split(":")[0])
    month = int(location["localtime"].split("-")[1])

    temp = current["temp_c"]
    humidity = current["humidity"]
    pressure = current["pressure_mb"]
    dew = current.get("dewpoint_c", temp)

    sample = {
        "lat": location["lat"],
        "lon": location["lon"],
        "temperature_C": temp,
        "humidity_pct": humidity,
        "pressure_hPa": pressure,
        "dew_point_C": dew,
        "solar_radiation_Wm2": current.get("uv", 5) * 100,
        "wind_speed_ms": current["wind_kph"] / 3.6,
        "wind_speed_kph": current["wind_kph"],
        "wind_direction_deg": wind_deg,
        "cloud_cover_pct": current["cloud"],
        "visibility_km": current.get("vis_km", 0),
        "uv_index": current.get("uv", 0),
        "precip_mm": current.get("precip_mm", 0),
        "feels_like_C": current.get("feelslike_c", temp),
        "wind_dir_sin": np.sin(np.radians(wind_deg)),
        "wind_dir_cos": np.cos(np.radians(wind_deg)),
        "et0_mm": 3,
        "pressure_trend": 0,
        "hour": hour,
        "month": month,
        "city_encoded": 1,
        "CO_ugm3": air.get("co", 0),
        "NO2_ugm3": air.get("no2", 0),
        "O3_ugm3": air.get("o3", 0),
        "SO2_ugm3": air.get("so2", 0),
        "PM2_5_ugm3": air.get("pm2_5", 0),
        "PM10_ugm3": air.get("pm10", 0)
    }

    sample["temp_dew_diff"] = temp - dew
    sample["humidity_pressure"] = humidity * pressure
    sample["pm_ratio"] = sample["PM2_5_ugm3"] / (sample["PM10_ugm3"] + 1)
    sample["gas_index"] = sample["NO2_ugm3"] + sample["SO2_ugm3"] + sample["O3_ugm3"]
    
    return sample, location

def calculate_heat_risk(temp, humidity, feels_like=None, ml_score=None):
    effective_temp = feels_like if feels_like is not None else temp
    if effective_temp >= 40:
        rule_score = 2
    elif effective_temp >= 32:
        rule_score = 1
    else:
        rule_score = 0

    if humidity >= 75 and effective_temp >= 30:
        rule_score = max(rule_score, 2)
    elif humidity >= 60 and effective_temp >= 28:
        rule_score = max(rule_score, 1)

    if ml_score is not None:
        final_score = max(rule_score, ml_score)
    else:
        final_score = rule_score

    if final_score == 0:
        level = "Low"
    elif final_score == 1:
        level = "Moderate"
    else:
        level = "High"

    return final_score, level

# =========================
# BUILD INPUT (Updated to float32)
# =========================
def build_input_dynamic(sample, model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = base_features

    values = [sample.get(col, 0) for col in features]
    expected = getattr(model, "n_features_in_", len(values))

    if len(values) > expected:
        values = values[:expected]
    elif len(values) < expected:
        values += [0] * (expected - len(values))

    return np.array([values], dtype=np.float32)
def get_lstm_features():

    scaler_features = getattr(
        lstm_scaler,
        "feature_names_in_",
        None
    )

    if scaler_features is not None:
        return list(scaler_features)

    config_features = lstm_config.get(
        "features",
        []
    )

    if len(config_features) == getattr(
        lstm_scaler,
        "n_features_in_",
        0
    ):
        return config_features

    return lstm_fallback_features[
        : getattr(
            lstm_scaler,
            "n_features_in_",
            len(lstm_fallback_features)
        )
    ]
# =========================
# INTERPRETATION
# =========================
def categorize_heat(score):
    if score == 0:
        return "Low"
    elif score == 1:
        return "Moderate"
    else:
        return "High"

def categorize_pollution(score):
    if score == 0:
        return "Good"
    elif score == 1:
        return "Moderate"
    elif score == 2:
        return "Unhealthy for Sensitive Groups"
    else:
        return "Unhealthy"

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"
def lstm_forecast(weather):

    try:

        lstm_features = get_lstm_features()

        precip_index = (
            lstm_features.index("precip_mm")
            if "precip_mm" in lstm_features
            else len(lstm_features) - 1
        )

        latest = np.array([[
            weather.get(feature, 0)
            for feature in lstm_features
        ]], dtype=np.float32)

        latest_scaled = lstm_scaler.transform(
            latest
        )

        sequence = deque(
            [latest_scaled[0]] * 7,
            maxlen=7
        )

        X_input = np.array(
            [sequence],
            dtype=np.float32
        )

        prediction = lstm_model.predict(
            X_input,
            verbose=0
        )[0]

        forecast = []

        for rain in prediction:

            dummy = latest_scaled.copy()

            dummy[0][precip_index] = rain

            original = lstm_scaler.inverse_transform(
                dummy
            )[0][precip_index]

            forecast.append(
                round(float(original), 2)
            )

        return forecast

    except Exception as e:

        return {
            "error": str(e)
        }
def lstm_forecast_rain_probabilities(
    weather,
    forecast_rainfall
):

    try:

        probabilities = []

        for rainfall_mm in forecast_rainfall:

            day_weather = weather.copy()

            day_weather["precip_mm"] = rainfall_mm

            X_rain = build_input_dynamic(
                day_weather,
                rain_model
            )

            rain_prob = float(
                rain_model.predict_proba(
                    X_rain
                )[0][1]
            )

            probabilities.append(
                round(rain_prob * 100, 2)
            )

        return probabilities

    except Exception:

        return [0.0, 0.0, 0.0]

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "Climate Intelligence API Running 🌍"}

@app.get("/predict/{query}")
def predict(query: str):
    try:
        weather, location = get_weather(query)

        # =====================================================
        # ENGINEERED FEATURES FOR V4 MODELS
        # =====================================================
        weather["moisture_flux"] = weather["humidity_pct"] * weather["wind_speed_ms"]
        weather["dewpoint_depression"] = weather["temperature_C"] - weather["dew_point_C"]
        weather["wind_humidity"] = weather["wind_speed_ms"] * weather["humidity_pct"]
        weather["cloud_dew"] = weather["cloud_cover_pct"] * weather["dew_point_C"]
        weather["cloud_humidity"] = weather["cloud_cover_pct"] * weather["humidity_pct"]

        # Lags & rolling features 
        weather["dew_spread_change"] = 0
        weather["solar_drop_3h"] = 0
        weather["pressure_change_1h"] = 0
        weather["pressure_change_3h"] = 0
        weather["pressure_change_6h"] = 0
        weather["humidity_change_1h"] = 0
        weather["humidity_change_3h"] = 0
        weather["temp_change_1h"] = 0
        weather["temp_change_3h"] = 0

        weather["precip_lag1"] = weather.get("precip_mm", 0)
        weather["precip_lag3"] = weather.get("precip_mm", 0)
        weather["humidity_lag1"] = weather.get("humidity_pct", 0)
        weather["pressure_lag1"] = weather.get("pressure_hPa", 0)
        weather["precip_roll3"] = weather.get("precip_mm", 0)
        weather["humidity_roll3"] = weather.get("humidity_pct", 0)
        weather["pressure_roll3"] = weather.get("pressure_hPa", 0)
        weather["rain_last_3h"] = weather.get("precip_mm", 0) * 3

        # Wind Vectors
        weather["wind_u"] = weather["wind_speed_ms"] * np.cos(np.radians(weather["wind_direction_deg"]))
        weather["wind_v"] = weather["wind_speed_ms"] * np.sin(np.radians(weather["wind_direction_deg"]))

        # Cyclical Time Encoding
        weather["month_sin"] = np.sin(2 * np.pi * weather["month"] / 12)
        weather["month_cos"] = np.cos(2 * np.pi * weather["month"] / 12)
        weather["hour_sin"] = np.sin(2 * np.pi * weather["hour"] / 24)
        weather["hour_cos"] = np.cos(2 * np.pi * weather["hour"] / 24)

        # Baseline Heuristic
        if weather["humidity_pct"] > 80 and weather["cloud_cover_pct"] > 70:
            weather["rain_tomorrow_loc_avg"] = 0.7
        else:
            weather["rain_tomorrow_loc_avg"] = 0.2

        # =====================================================
        # MODEL INFERENCES
        # =====================================================
        
        # 🌧️ Rain (Fully Updated logic)
        rain_features = list(rain_model.feature_names_in_)
        X_rain = np.array([[weather.get(f, 0) for f in rain_features]], dtype=np.float32)
        
        rain_prob = float(rain_model.predict_proba(X_rain)[0][1])
        rain_percent = round(rain_prob * 100, 2)
        rain_pred = int(rain_prob > rain_threshold)

        # ⚡ Storm
        X_storm = build_input_dynamic(weather, storm_model)
        storm_prob = float(storm_model.predict_proba(X_storm)[0][1])

        # 🌡️ Heat
        X_heat = build_input_dynamic(weather, heat_model)
        temp = weather["temperature_C"]
        humidity = weather["humidity_pct"]
        feels_like = weather.get("feels_like_C")

        ml_heat_score = float(heat_model.predict(X_heat)[0])

        heat_score, heat_level = calculate_heat_risk(
            temp=temp,
            humidity=humidity,
            feels_like=feels_like,
            ml_score=ml_heat_score
        )

        # 🔥 Safety overrides
        if temp < 26:
            heat_score = 0
        elif temp >= 30:
            heat_score = max(heat_score, 1)

        if feels_like is not None and feels_like >= 35:
            heat_score = max(heat_score, 2)

        heat_level = categorize_heat(heat_score)

        # 🌫️ Pollution
        try:
            X_pollution = build_input_dynamic(weather, pollution_model)
            pollution_score = float(pollution_model.predict(X_pollution)[0])
        except Exception as e:
            print("Pollution prediction error:", e)
            pm25 = weather.get("PM2_5_ugm3", 0)
            pm10 = weather.get("PM10_ugm3", 0)
            o3 = weather.get("O3_ugm3", 0)
            if pm25 > 55 or pm10 > 150 or o3 > 180:
                pollution_score = 2
            elif pm25 > 25 or pm10 > 60 or o3 > 120:
                pollution_score = 1
            else:
                pollution_score = 0          
        
        # AQI Model
        X_aqi = np.array([[
            weather["temperature_C"],
            weather["humidity_pct"],
            weather["wind_speed_ms"],
            weather["cloud_cover_pct"],
            weather["hour"],
            weather["month"]
        ]], dtype=np.float32)

        predicted_aqi = float(aqi_model.predict(X_aqi)[0])
        aqi_category = categorize_aqi(predicted_aqi)
# =========================
# LSTM FORECAST
# =========================

        future_forecast = lstm_forecast(weather)

        if isinstance(future_forecast, dict):

            lstm_response = future_forecast

        else:

            future_probabilities = (
                lstm_forecast_rain_probabilities(
                    weather,
                    future_forecast
                )
            )

            lstm_response = {

                "next_3_days_rainfall_mm":
                    future_forecast,

                "next_3_days_rain_probability_%":
                    future_probabilities
            }
        # =====================================================
        # RETURN RESPONSE
        # =====================================================
        return {
            "query": query,
            "resolved_location": {
                "region": location["region"],
                "country": location["country"]
            },
            "location": {
                "lat": weather["lat"],
                "lon": weather["lon"]
            },
            "current_weather": {
                "temperature_C": weather["temperature_C"],
                "feels_like_C": weather["feels_like_C"],
                "humidity_pct": weather["humidity_pct"],
                "pressure_hPa": weather["pressure_hPa"],
                "cloud_cover_pct": weather["cloud_cover_pct"],
                "wind": {
                    "speed_ms": round(weather["wind_speed_ms"], 2),
                    "speed_kph": round(weather["wind_speed_kph"], 2),
                    "direction_deg": weather["wind_direction_deg"],
                    "direction": get_wind_direction(weather["wind_direction_deg"])
                },
                "visibility_km": weather["visibility_km"],
                "uv_index": weather["uv_index"],
                "precip_mm": weather["precip_mm"],
                "solar_radiation_Wm2": weather["solar_radiation_Wm2"],
                "dew_point_C": weather["dew_point_C"]
            },
            "rain": {
                "probability_%": rain_percent,
                "prediction": rain_pred
            },
            "thunderstorm": {
                "probability_%": round(storm_prob * 100, 2)
            },
            "heat_risk": {
                "score": round(heat_score, 2),
                "level": heat_level
            }, 
            "air_pollution": {
                "score": round(pollution_score, 2),
                "category": categorize_pollution(pollution_score)
            },
            "aqi_prediction": {
                "aqi": round(predicted_aqi),
                "category": aqi_category
            },
            "lstm_forecast": lstm_response,
            "air_gases": {
                "co": round(weather["CO_ugm3"], 2),
                "no2": round(weather["NO2_ugm3"], 2),
                "o3": round(weather["O3_ugm3"], 2),
                "so2": round(weather["SO2_ugm3"], 2),
                "pm2_5": round(weather["PM2_5_ugm3"], 2),
                "pm10": round(weather["PM10_ugm3"], 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}
#python -m uvicorn api.main:app --reload
