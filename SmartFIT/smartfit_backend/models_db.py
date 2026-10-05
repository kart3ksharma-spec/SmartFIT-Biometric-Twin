"""
SmartFit Backend - Database Models
"""
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship: one user can have many logged profile/prediction entries
    predictions = db.relationship('PredictionLog', backref='user', lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {"id": self.id, "username": self.username, "email": self.email}


class PredictionLog(db.Model):
    """Stores each prediction request so the Goal Tracking / Digital Twin
    modules can later show progress over time."""
    __tablename__ = 'prediction_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Input snapshot
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    bmi = db.Column(db.Float)
    session_duration = db.Column(db.Float)
    avg_bpm = db.Column(db.Float)
    resting_bpm = db.Column(db.Float)
    water_intake = db.Column(db.Float)
    workout_frequency = db.Column(db.Integer)
    fat_percentage = db.Column(db.Float)
    calories_burned = db.Column(db.Float)

    # Prediction output snapshot
    predicted_experience_level = db.Column(db.Integer)
    recommended_workout = db.Column(db.String(50))
    predicted_calories_burned = db.Column(db.Float)
    fitness_persona = db.Column(db.String(100))

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "predicted_experience_level": self.predicted_experience_level,
            "recommended_workout": self.recommended_workout,
            "predicted_calories_burned": self.predicted_calories_burned,
            "fitness_persona": self.fitness_persona,
        }
