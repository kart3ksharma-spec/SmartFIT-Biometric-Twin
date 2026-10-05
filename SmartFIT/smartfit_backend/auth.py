"""
SmartFit Backend - JWT Authentication Helpers
"""
import jwt
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app

SECRET_KEY = os.environ.get('SMARTFIT_SECRET_KEY', 'dev-secret-key-change-in-production')
TOKEN_EXPIRY_HOURS = 24


def generate_token(user_id: int) -> str:
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        'iat': datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def decode_token(token: str):
    """Returns user_id if valid, raises jwt exceptions if not."""
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    return payload['user_id']


def token_required(f):
    """Decorator to protect routes -- requires a valid JWT in the
    Authorization header: 'Authorization: Bearer <token>'"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(' ', 1)[1]
        try:
            user_id = decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(current_user_id=user_id, *args, **kwargs)
    return decorated
