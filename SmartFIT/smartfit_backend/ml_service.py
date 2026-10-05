"""
SmartFit Backend - ML Prediction Service
Loads all trained models once at startup and exposes a single function
to run the full prediction pipeline for a given user profile.
"""
import pandas as pd
import joblib
import os
from SmartFIT.smartfit_backend.workout_rules import recommend_workout

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _load(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))


# ---- Load all models & encoders once at import time ----
clf_model = _load('classification_model.pkl')
clf_gender_enc = _load('gender_encoder.pkl')
clf_scaler = _load('classification_scaler.pkl')
clf_features = _load('classification_features.pkl')
clf_best_name = _load('classification_best_model_name.pkl')

reg_model = _load('regression_model.pkl')
reg_gender_enc = _load('regression_gender_encoder.pkl')
reg_workout_enc = _load('regression_workout_type_encoder.pkl')
reg_scaler = _load('regression_scaler.pkl')
reg_features = _load('regression_features.pkl')
reg_best_name = _load('regression_best_model_name.pkl')

cluster_model = _load('clustering_model.pkl')
cluster_scaler = _load('clustering_scaler.pkl')
cluster_features = _load('clustering_features.pkl')
persona_labels = _load('persona_labels.pkl')


REQUIRED_FIELDS = [
    "age", "gender", "bmi", "session_duration", "avg_bpm", "resting_bpm",
    "water_intake", "workout_frequency", "fat_percentage", "calories_burned",
]


def validate_user_input(user: dict):
    """Returns a list of missing/invalid fields, empty list if valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in user or user[field] in (None, ""):
            errors.append(f"Missing field: {field}")
    if "gender" in user and user["gender"] not in ("Male", "Female"):
        errors.append("gender must be 'Male' or 'Female'")
    return errors


def predict_experience_level(user: dict) -> int:
    row = pd.DataFrame([{
        'Age': user['age'], 'Gender': user['gender'], 'BMI': user['bmi'],
        'Session_Duration (hours)': user['session_duration'], 'Avg_BPM': user['avg_bpm'],
        'Resting_BPM': user['resting_bpm'], 'Water_Intake (liters)': user['water_intake'],
        'Workout_Frequency (days/week)': user['workout_frequency'],
        'Fat_Percentage': user['fat_percentage'], 'Calories_Burned': user['calories_burned'],
    }])
    row['Gender'] = clf_gender_enc.transform(row['Gender'])
    row = row[clf_features]
    if clf_best_name == "Random Forest":
        pred = clf_model.predict(row)[0]
    else:
        pred = clf_model.predict(clf_scaler.transform(row))[0]
    return int(pred)


def predict_calories(user: dict, workout_type: str, experience_level: int) -> float:
    row = pd.DataFrame([{
        'Age': user['age'], 'Gender': user['gender'], 'BMI': user['bmi'],
        'Session_Duration (hours)': user['session_duration'], 'Avg_BPM': user['avg_bpm'],
        'Resting_BPM': user['resting_bpm'], 'Water_Intake (liters)': user['water_intake'],
        'Workout_Frequency (days/week)': user['workout_frequency'],
        'Workout_Type': workout_type, 'Experience_Level': experience_level,
    }])
    row['Gender'] = reg_gender_enc.transform(row['Gender'])
    row['Workout_Type'] = reg_workout_enc.transform(row['Workout_Type'])
    row = row[reg_features]
    if reg_best_name == "Random Forest Regressor":
        pred = reg_model.predict(row)[0]
    else:
        pred = reg_model.predict(reg_scaler.transform(row))[0]
    return round(float(pred), 1)


def predict_persona(user: dict) -> str:
    row = pd.DataFrame([{
        'Age': user['age'], 'BMI': user['bmi'],
        'Session_Duration (hours)': user['session_duration'], 'Avg_BPM': user['avg_bpm'],
        'Fat_Percentage': user['fat_percentage'],
        'Workout_Frequency (days/week)': user['workout_frequency'],
        'Calories_Burned': user['calories_burned'],
    }])
    row = row[cluster_features]
    cluster_id = cluster_model.predict(cluster_scaler.transform(row))[0]
    return persona_labels[cluster_id]


def run_smartfit_pipeline(user: dict) -> dict:
    """Full SmartFit prediction pipeline for one user profile.
    Raises ValueError if input is invalid."""
    errors = validate_user_input(user)
    if errors:
        raise ValueError("; ".join(errors))

    experience_level = predict_experience_level(user)
    workout_rec = recommend_workout(experience_level)
    predicted_calories = predict_calories(user, workout_rec['primary'], experience_level)
    persona = predict_persona(user)

    return {
        "predicted_experience_level": experience_level,
        "experience_label": {1: "Beginner", 2: "Intermediate", 3: "Advanced"}[experience_level],
        "recommended_workout": workout_rec['primary'],
        "secondary_workout": workout_rec['secondary'],
        "reasoning": workout_rec['reasoning'],
        "predicted_calories_burned": predicted_calories,
        "fitness_persona": persona,
    }
