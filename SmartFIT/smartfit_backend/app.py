"""
SmartFit Backend - Main Flask Application

Routes:
  POST /api/register        - create a new user account
  POST /api/login           - log in, returns JWT token
  GET  /api/me               - get current logged-in user's info (protected)
  POST /api/predict          - run the full ML pipeline for a user profile (protected)
  GET  /api/history          - get the logged-in user's past predictions (protected)
  GET  /api/health           - simple health check (no auth)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

from SmartFIT.smartfit_backend.models_db import db, User, PredictionLog
from SmartFIT.smartfit_backend.auth import generate_token, token_required
from SmartFIT.smartfit_backend.ml_service import run_smartfit_pipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smartfit.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
CORS(app)  # allows the React frontend (different port) to call this API

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "SmartFit API"})


# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400
    if not EMAIL_REGEX.match(email):
        return jsonify({"error": "Invalid email format"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    token = generate_token(user.id)
    return jsonify({"message": "Account created", "token": token, "user": user.to_dict()}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    token = generate_token(user.id)
    return jsonify({"message": "Login successful", "token": token, "user": user.to_dict()})


@app.route('/api/me', methods=['GET'])
@token_required
def me(current_user_id):
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())


# ---------------------------------------------------------------------------
# PREDICTION (core ML pipeline)
# ---------------------------------------------------------------------------

@app.route('/api/predict', methods=['POST'])
@token_required
def predict(current_user_id):
    data = request.get_json(silent=True) or {}

    try:
        result = run_smartfit_pipeline(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Log this prediction so Goal Tracking / history can use it later
    log = PredictionLog(
        user_id=current_user_id,
        age=data.get('age'), gender=data.get('gender'), bmi=data.get('bmi'),
        session_duration=data.get('session_duration'), avg_bpm=data.get('avg_bpm'),
        resting_bpm=data.get('resting_bpm'), water_intake=data.get('water_intake'),
        workout_frequency=data.get('workout_frequency'), fat_percentage=data.get('fat_percentage'),
        calories_burned=data.get('calories_burned'),
        predicted_experience_level=result['predicted_experience_level'],
        recommended_workout=result['recommended_workout'],
        predicted_calories_burned=result['predicted_calories_burned'],
        fitness_persona=result['fitness_persona'],
    )
    db.session.add(log)
    db.session.commit()

    result['log_id'] = log.id
    return jsonify(result)


@app.route('/api/history', methods=['GET'])
@token_required
def history(current_user_id):
    logs = (PredictionLog.query
            .filter_by(user_id=current_user_id)
            .order_by(PredictionLog.timestamp.desc())
            .limit(50)
            .all())
    return jsonify([log.to_dict() for log in logs])


# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, port=5000)
