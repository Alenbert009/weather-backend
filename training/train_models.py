import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
df = pd.read_csv("data/processed/final_dataset.csv")
df_low = df[df['Heat Risk'] == 0].sample(2000000)
df_med = df[df['Heat Risk'] == 1]
df_high = df[df['Heat Risk'] == 2]

df_heat_balanced = pd.concat([df_low, df_med, df_high])
df_heat_balanced['heat_index'] = df_heat_balanced['temperature_C'] * (df_heat_balanced['humidity_pct'] / 100)
print(df_heat_balanced['Heat Risk'].value_counts())
# # #  HEAT MODEL
X_heat = df_heat_balanced[['temperature_C', 'humidity_pct', 'hour','wind_speed_ms','cloud_cover_pct','heat_index']]
y_heat = df_heat_balanced['Heat Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X_heat, y_heat, test_size=0.2, random_state=42,stratify=y_heat
)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
heat_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    reg_alpha=1, #L1 Regularization
    reg_lambda=2,#L2 Regularization
    objective='multi:softprob',  # important
    num_class=3,
    eval_metric='mlogloss',
    n_jobs=-1
)

heat_model.fit(X_train, y_train,sample_weight=sample_weights,
               eval_set=[(X_test, y_test)],
                verbose=False)
heat_probs = heat_model.predict_proba(X_test)
heat_pred = np.argmax(heat_probs, axis=1)

print(classification_report(y_test, heat_pred))
print("\nHeat Model Accuracy:", accuracy_score(y_test, heat_pred))
joblib.dump(heat_model,'models/heat_model.pkl')

#  THUNDERSTORM MODEL
X_thunder = df[
    ['humidity_pct',
     'cloud_cover_pct',
     'wind_speed_ms',
     'hour',
     'pressure_hPa',
     'pressure_trend']
]
y_thunder = df['thunderstorm_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X_thunder, y_thunder, test_size=0.2, random_state=42,stratify=y_thunder
)
# Calculate imbalance ratio
neg, pos = y_train.value_counts()
scale_weight = neg / pos
print(f"Imbalance ratio (neg/pos): {scale_weight:.2f}")
thunder_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    reg_alpha=1,
    reg_lambda=1,
    eval_metric='logloss',
    n_jobs=-1
)


thunder_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                    verbose=False)

thunder_probs = thunder_model.predict_proba(X_test)
thunder_pred =  np.argmax(thunder_probs, axis=1)
print("\nThunderstorm Model Accuracy:", accuracy_score(y_test, thunder_pred))
print(classification_report(y_test, thunder_pred))
joblib.dump(thunder_model, "models/thunderstorm_model.pkl")


# #  POLLUTION MODEL
df['pm_ratio'] = df['PM2_5_ugm3'] / (df['PM10_ugm3'] + 1)
df['gas_index'] = df['NO2_ugm3'] + df['SO2_ugm3'] + df['O3_ugm3']
X_poll = df[[
    'PM2_5_ugm3','PM10_ugm3',
    'NO2_ugm3','CO_ugm3','SO2_ugm3','O3_ugm3',
    'humidity_pct','wind_speed_ms','month','hour',
    'pm_ratio','gas_index'
]]
y_poll = df['pollution_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X_poll, y_poll, test_size=0.2, random_state=42,stratify=y_poll
)
# Handle imbalance
poll_weights = compute_sample_weight(class_weight='balanced', y=y_train)

poll_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    reg_alpha=1,
    reg_lambda=1,
    objective='multi:softprob',  # important
    num_class=3,
    eval_metric='mlogloss',
    n_jobs=-1
)

poll_model.fit(X_train, y_train, sample_weight=poll_weights,
               eval_set=[(X_test, y_test)],
                verbose=False)
poll_pred = np.argmax(poll_model.predict_proba(X_test), axis=1)

print(" Pollution Model Accuracy:", accuracy_score(y_test, poll_pred))
print(classification_report(y_test, poll_pred))
joblib.dump(poll_model, "models/pollution_model.pkl")
print("\n All models trained and saved successfully!")