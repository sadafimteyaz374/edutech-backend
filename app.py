from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pickle
import pandas as pd
import numpy as np
import os
import hashlib
import datetime

app = Flask(__name__)
# Updated CORS to handle all standard headers
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# MONGODB ATLAS CONNECTION
MONGO_URI = "mongodb+srv://SS_Edu:S%40123@cluster0.rnne2al.mongodb.net/?retryWrites=true&w=majority"

db = None
users_collection = None
predictions_collection = None
contacts_collection = None

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping') 
    
    db = client["EduTech_DB"]
    users_collection = db["users"]
    predictions_collection = db["predictions"]
    contacts_collection = db["contacts"] # <--- Initialize Contacts
    print("✓ MongoDB Atlas connected | DB: EduTech_DB")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    
# LOAD ML ASSETS
base_dir  = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'models')

try:
    model             = pickle.load(open(os.path.join(model_dir, 'best_model.pkl'), 'rb'))
    scaler            = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'),     'rb'))
    selected_features = pickle.load(open(os.path.join(model_dir, 'features.pkl'),   'rb'))
    print(f"✓ ML assets loaded | Features ({len(selected_features)}): {selected_features}")
except Exception as e:
    print(f"✗ Error loading ML assets: {e}")
    model = scaler = None
    selected_features = []

# HELPERS
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

@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        message = data.get('message', '').strip()

        if not all([name, email, message]):
            return jsonify({'error': 'Name, email and message are required'}), 400

        # Save to MongoDB
        contact_record = {
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        contacts_collection.insert_one(contact_record)
        
        print(f"✓ Contact message saved from: {email}")
        return jsonify({'message': 'Message sent successfully! We will get back to you.'}), 201

    except Exception as e:
        print(f"✗ Contact error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    db_status = "connected" if client else "disconnected"
    return jsonify({'status': 'ok', 'database': db_status, 'features': selected_features})

@app.route('/api/register', methods=['POST'])
def register():
    data     = request.get_json()
    name     = data.get('name',     '').strip()
    email     = data.get('email',    '').strip().lower()
    password = data.get('password', '')
    course   = data.get('course',   '').strip()
    year     = data.get('year',     '').strip()

    if not all([name, email, password]):
        return jsonify({'error': 'Name, email and password are required'}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({'error': 'Email already registered'}), 409

    users_collection.insert_one({
        'name':          name,
        'email':         email,
        'password_hash': hash_password(password),
        'course':        course,
        'year':          year,
        'created_at':    datetime.datetime.utcnow().isoformat()
    })

    token = make_token(email)
    active_tokens[token] = email

    return jsonify({
        'message': 'Registration successful',
        'token':   token,
        'user':    {'name': name, 'email': email, 'course': course, 'year': year}
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    email     = data.get('email',    '').strip().lower()
    password = data.get('password', '')

    user = users_collection.find_one({"email": email})
    if not user or user['password_hash'] != hash_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = make_token(email)
    active_tokens[token] = email

    return jsonify({
        'message': 'Login successful',
        'token':   token,
        'user': {
            'name':   user['name'],
            'email':  user['email'],
            'course': user['course'],
            'year':   user['year']
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    email = get_current_user(request)
    if not email:
        return jsonify({'error': 'Unauthorized. Please login.'}), 401

    data = request.get_json()
    student_name = data.get('student_name', 'Unknown').strip()

    try:
        feature_values = {}
        for feat in selected_features:
            val = data.get(feat)
            if val is None:
                return jsonify({'error': f'Missing feature: {feat}'}), 400
            feature_values[feat] = float(val)

        input_df = pd.DataFrame([list(feature_values.values())], columns=selected_features)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        
        label = 'Pass' if int(pred) == 1 else 'Fail'
        
        history_record = {
            'email': email,
            'student_name': student_name,
            'prediction': label,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        history_record.update(feature_values)

        predictions_collection.insert_one(history_record)
        return jsonify({'prediction': label, 'student_name': student_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def history():
    email = get_current_user(request)
    if not email:
        return jsonify({'error': 'Unauthorized'}), 401

    records = list(predictions_collection.find(
        {"email": email},
        {"_id": 0, "email": 0}
    ).sort("timestamp", -1))
    
    return jsonify(records)

@app.route('/api/features', methods=['GET'])
def get_features():
    feature_meta = {
        'G1':        {'label': 'First Period Grade (G1)',       'min': 0,  'max': 20, 'step': 1, 'type': 'number'},
        'failures':  {'label': 'Past Class Failures',           'min': 0,  'max': 4,  'step': 1, 'type': 'number'},
        'goout':     {'label': 'Going Out with Friends (1-5)',  'min': 1,  'max': 5,  'step': 1, 'type': 'number'},
        'age':       {'label': 'Student Age',                   'min': 15, 'max': 22, 'step': 1, 'type': 'number'},
        'higher':    {'label': 'Wants Higher Education',        'type': 'select',
                      'options': [{'value': 1, 'label': 'Yes'}, {'value': 0, 'label': 'No'}]},
        'Medu':      {'label': "Mother's Education (0-4)",      'min': 0,  'max': 4,  'step': 1, 'type': 'number'},
        'Fedu':      {'label': "Father's Education (0-4)",      'min': 0,  'max': 4,  'step': 1, 'type': 'number'},
        'guardian':  {'label': 'Guardian',                      'type': 'select',
                      'options': [{'value': 0, 'label': 'Father'}, {'value': 1, 'label': 'Mother'}, {'value': 2, 'label': 'Other'}]},
        'schoolsup': {'label': 'Extra School Support',          'type': 'select',
                      'options': [{'value': 1, 'label': 'Yes'}, {'value': 0, 'label': 'No'}]},
        'reason':    {'label': 'Reason to Choose School (0-3)', 'min': 0,  'max': 3,  'step': 1, 'type': 'number'},
        'romantic':  {'label': 'In a Romantic Relationship',    'type': 'select',
                      'options': [{'value': 1, 'label': 'Yes'}, {'value': 0, 'label': 'No'}]},
        'paid':      {'label': 'Extra Paid Classes',            'type': 'select',
                      'options': [{'value': 1, 'label': 'Yes'}, {'value': 0, 'label': 'No'}]},
        'absences':  {'label': 'Number of School Absences',     'min': 0,  'max': 93, 'step': 1, 'type': 'number'},
    }
    return jsonify({'features': selected_features, 'meta': feature_meta})

#ADMIN DASHBOARD ROUTES

@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    try:
        # Fetch all users, exclude password hashes for security
        users = list(users_collection.find({}, {"_id": 0, "password_hash": 0}))
        return jsonify({"users": users}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/contacts', methods=['GET'])
def admin_get_contacts():
    try:
        contacts = list(contacts_collection.find({}, {"_id": 0}))
        return jsonify({"contacts": contacts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/predictions', methods=['GET'])
def admin_get_predictions():
    try:
        # Get history records and convert ObjectId to string 
        predictions = list(predictions_collection.find({}))
        for p in predictions:
            p['_id'] = str(p['_id'])
            # Ensure keys match React
            p['result'] = p.get('prediction', 'N/A')
            p['accuracy'] = 82.28  # Displaying your Random Forest accuracy
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/delete_user', methods=['POST'])
def admin_delete_user():
    try:
        data = request.get_json()
        email = data.get('email')
        users_collection.delete_one({"email": email})
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/delete_contact', methods=['POST'])
def admin_delete_contact():
    try:
        data = request.get_json()
        email = data.get('email')
        timestamp = data.get('timestamp')
        contacts_collection.delete_one({"email": email, "timestamp": timestamp})
        return jsonify({"message": "Message removed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/delete_prediction', methods=['POST'])
def admin_delete_prediction():
    try:
        from bson import ObjectId
        data = request.get_json()
        pred_id = data.get('id')
        predictions_collection.delete_one({"_id": ObjectId(pred_id)})
        return jsonify({"message": "Prediction cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)