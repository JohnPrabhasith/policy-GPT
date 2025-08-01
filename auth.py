import os
from flask import request, jsonify

def verify_token():
    """
    Flask function to verify the Bearer token from the Authorization header.
    Returns True if valid, otherwise returns a Flask response object with an error.
    """
    expected_token = os.getenv("API_BEARER_TOKEN")
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Invalid or missing Authorization header. Use 'Bearer <token>'."}), 401

    token = auth_header.split(" ")[1]

    if token != expected_token:
        return jsonify({"error": "Invalid or expired token!"}), 403

    return True  # Token is valid
