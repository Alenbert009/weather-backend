from tensorflow.keras.models import load_model

model = load_model(
    "training/weather_lstm_model.h5",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")
model.summary()