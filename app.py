from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pickle
import pandas as pd
import numpy as np
import os
import hashlib
import datetime
import certifi  


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]}})

# --- DATABASE CONFIGURATION ---
MONGO_URI = "mongodb+srv://SS_Edu:S%40123@cluster0.rnne2al.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(
        MONGO_URI, 
        serverSelectionTimeoutMS=5000,
        tlsCAFile=certifi.where() # Atlas connection ko verify karne ke liye
    )
    client.server_info()  # Test connection
    db = client["EduTech_DB"]
    
    # Collections setup
    users_collection = db["users"]
    predictions_collection = db["predictions"]
    contacts_collection = db["contacts"]
    
    print("✓ MongoDB Atlas connected successfully | DB: EduTech_DB")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    client = None

# --- ML ASSETS LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'models')

try:
    model = pickle.load(open(os.path.join(model_dir, 'best_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    selected_features = pickle.load(open(os.path.join(model_dir, 'features.pkl'), 'rb'))
    print(f"✓ ML assets loaded | Features: {selected_features}")
except Exception as e:
    print(f"✗ Error loading ML assets: {e}")
    model = scaler = None
    selected_features = []

# --- UTILITY FUNCTIONS ---
def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def make_token(email):
    raw = f"{email}:{datetime.datetime.utcnow().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()

active_tokens = {}

def get_current_user(req):
    auth = req.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:]
        return active_tokens.get(token)
    return None

# --- ROUTES ---

@app.route('/api/health', methods=['GET'])
def health():
    db_status = "connected" if client and client.server_info() else "disconnected"
    return jsonify({'status': 'ok', 'database': db_status})

# REGISTER
@app.route('/api/register', methods=['POST'])
def register():
    if not client: return jsonify({'error': 'Database connection down'}), 500
    data = request.get_json()
    name, email = data.get('name', '').strip(), data.get('email', '').strip().lower()
    password = data.get('password', '')
    course, year = data.get('course', '').strip(), data.get('year', '').strip()

    if not all([name, email, password]):
        return jsonify({'error': 'All fields are required'}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({'error': 'Email already registered'}), 409

    users_collection.insert_one({
        'name': name, 'email': email,
        'password_hash': hash_password(password),
        'course': course, 'year': year,
        'created_at': datetime.datetime.utcnow().isoformat()
    })

    token = make_token(email)
    active_tokens[token] = email
    return jsonify({'message': 'Success', 'token': token, 'user': {'name': name, 'email': email}}), 201

# LOGIN
@app.route('/api/login', methods=['POST'])
def login():
    if not client: return jsonify({'error': 'Database connection down'}), 500
    data = request.get_json()
    email, password = data.get('email', '').strip().lower(), data.get('password', '')

    user = users_collection.find_one({"email": email})
    if not user or user['password_hash'] != hash_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = make_token(email)
    active_tokens[token] = email
    return jsonify({
        'token': token, 
        'user': {'name': user['name'], 'email': user['email'], 'course': user['course'], 'year': user['year']}
    })

# CONTACT
@app.route('/api/contact', methods=['POST'])
def contact():
    if not client: return jsonify({'error': 'Database connection down'}), 500
    data = request.get_json()
    name, email, message = data.get('name', ''), data.get('email', ''), data.get('message', '')

    try:
        contacts_collection.insert_one({
            'name': name, 'email': email, 'message': message,
            'timestamp': datetime.datetime.utcnow().isoformat()
        })
        return jsonify({'message': 'Sent'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# PREDICT
@app.route('/api/predict', methods=['POST'])
def predict():
    email = get_current_user(request)
    if not email: return jsonify({'error': 'Please login first'}), 401
    if model is None: return jsonify({'error': 'ML Model not found'}), 500

    data = request.get_json()
    student_name = data.get('student_name', 'Student')

    try:
        feature_values = {feat: float(data.get(feat, 0)) for feat in selected_features}
        input_df = pd.DataFrame([list(feature_values.values())], columns=selected_features)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        
        label = 'Pass' if int(pred) == 1 else 'Fail'
        
        predictions_collection.insert_one({
            'email': email, 'student_name': student_name,
            'prediction': label, 'timestamp': datetime.datetime.utcnow().isoformat()
        })

        return jsonify({'prediction': label, 'student_name': student_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#ADMIN ROUTES

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        users = list(users_collection.find({}, {"_id": 0, "password_hash": 0}))
        return jsonify({'users': users}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/contacts', methods=['GET'])
def get_all_contacts():
    try:
        contacts = list(contacts_collection.find({}, {"_id": 0}))
        return jsonify({'contacts': contacts}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# USER DELETE ROUTE
@app.route('/api/admin/delete_user/<email>', methods=['DELETE'])
def delete_user(email):
    try:
        result = users_collection.delete_one({"email": email})
        if result.deleted_count > 0:
            return jsonify({'message': f'User {email} deleted successfully'}), 200
        return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# CONTACT/COMMENT DELETE ROUTE
@app.route('/api/admin/delete_contact', methods=['POST'])
def delete_contact():
    data = request.get_json()
    email = data.get('email')
    timestamp = data.get('timestamp')
    
    try:
        # Email aur timestamp dono use kar rahe hain taaki sahi comment delete ho
        result = contacts_collection.delete_one({"email": email, "timestamp": timestamp})
        if result.deleted_count > 0:
            return jsonify({'message': 'Comment deleted successfully'}), 200
        return jsonify({'error': 'Comment not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)