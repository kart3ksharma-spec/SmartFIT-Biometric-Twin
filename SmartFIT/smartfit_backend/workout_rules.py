"""
SmartFit - Step 5: Rule-Based Workout Recommendation Module
This layer sits on top of the verified Experience_Level classification model.

NOTE ON DESIGN DECISION:
We tested whether Workout_Type (Cardio/Strength/Yoga/HIIT) could be predicted
directly from a user's physical profile. It could not -- accuracy was ~23-33%,
statistically no better than random guessing, across two different datasets.
This makes sense: workout preference is driven by personal choice, goals, and
availability, not just body metrics.

Instead, we use a genuinely learnable ML output -- Experience_Level (90% model
accuracy) -- and combine it with simple, transparent rules (the kind a real
trainer would use) to produce a workout recommendation. This is an honest,
defensible design: ML where there is real signal, rules where there isn't.
"""

WORKOUT_RULES = {
    1: {  # Beginner
        "primary": "Cardio",
        "secondary": "Light Strength Training",
        "reasoning": "Beginners benefit most from building cardiovascular "
                      "endurance and basic movement patterns before progressing "
                      "to higher-intensity training."
    },
    2: {  # Intermediate
        "primary": "Strength Training",
        "secondary": "Moderate Cardio",
        "reasoning": "Intermediate users have established a training habit and "
                      "benefit from building strength alongside continued "
                      "cardiovascular work."
    },
    3: {  # Advanced
        "primary": "HIIT",
        "secondary": "Strength Training",
        "reasoning": "Advanced users have the conditioning to handle high-intensity "
                      "interval training safely, combined with strength work for "
                      "continued progress."
    },
}


def recommend_workout(experience_level: int) -> dict:
    """Given a predicted Experience_Level (1, 2, or 3), return a workout
    recommendation with reasoning."""
    return WORKOUT_RULES.get(experience_level, WORKOUT_RULES[1])


if __name__ == "__main__":
    for level in [1, 2, 3]:
        rec = recommend_workout(level)
        print(f"\nExperience_Level {level}:")
        print(f"  Primary recommendation:   {rec['primary']}")
        print(f"  Secondary recommendation: {rec['secondary']}")
        print(f"  Reasoning: {rec['reasoning']}")
