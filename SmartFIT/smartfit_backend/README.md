# SmartFit Backend (Flask API)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Server starts at `http://127.0.0.1:5000`

## Endpoints

| Method | Endpoint         | Auth? | Description |
|--------|------------------|-------|--------------|
| GET    | /api/health      | No    | Health check |
| POST   | /api/register    | No    | Create account. Body: `{username, email, password}` |
| POST   | /api/login       | No    | Log in. Body: `{username, password}`. Returns JWT token |
| GET    | /api/me          | Yes   | Get current user info |
| POST   | /api/predict     | Yes   | Run the full ML pipeline. See body format below |
| GET    | /api/history     | Yes   | Get logged-in user's past predictions |

For protected routes, send the JWT in the header:
`Authorization: Bearer <token>`

## /api/predict request body

```json
{
  "age": 32,
  "gender": "Female",
  "bmi": 27.0,
  "session_duration": 0.8,
  "avg_bpm": 145,
  "resting_bpm": 68,
  "water_intake": 2.3,
  "workout_frequency": 3,
  "fat_percentage": 29.0,
  "calories_burned": 700
}
```

## /api/predict response

```json
{
  "predicted_experience_level": 1,
  "experience_label": "Beginner",
  "recommended_workout": "Cardio",
  "secondary_workout": "Light Strength Training",
  "reasoning": "Beginners benefit most from building cardiovascular endurance...",
  "predicted_calories_burned": 569.3,
  "fitness_persona": "Beginner / Low Engagement",
  "log_id": 1
}
```

## Notes

- Uses SQLite (`smartfit.db`, auto-created on first run) for easy local development.
  Swap `SQLALCHEMY_DATABASE_URI` in `app.py` for a PostgreSQL URI when deploying.
- Models are pre-trained and loaded from the `models/` folder (see the
  `smartfit_ml_pipeline` project for the training scripts).
- `workout_rules.py` contains the rule-based recommendation logic layered on
  top of the ML-predicted Experience_Level (see project notes on why
  Workout_Type itself is not directly predicted).
