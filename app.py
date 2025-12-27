from flask import Flask, render_template, redirect, url_for, request, session
import joblib
import numpy as np
import pandas as pd
from flask import flash
import sqlite3
from functools import wraps
import os

app = Flask(__name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
app.secret_key = os.environ.get('SECRET_KEY', 'your_super_secret_key_12345')  # Set this to a strong, random value in production

# Database initialization function
def init_db():
    conn = sqlite3.connect('student_performance.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback TEXT,
            comments TEXT,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create contact table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create chat table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('register'))
    return render_template('index.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Connect to SQLite
        
        conn = sqlite3.connect('student_performance.db')
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is not None:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Check if email already exists
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone() is not None:
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        
        # Insert new user
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)  # In production, hash the password!
            )
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as err:
            flash('An error occurred. Please try again.', 'danger')
            return redirect(url_for('register'))
        finally:
            cursor.close()
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Connect to SQLite
        conn = sqlite3.connect('student_performance.db')
        cursor = conn.cursor()
        
        # Check credentials
        cursor.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)  # In production, verify hashed password!
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'danger')
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/services')
@login_required
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Connect to SQLite
        conn = sqlite3.connect('student_performance.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO contact_messages (name, email, message) VALUES (?, ?, ?)",
            (name, email, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
        flash('Thank you for contacting us! We have received your message.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')



# --- Prediction Route ---
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out!', 'success')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Collect form data
        input_data = {
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Previous GPA': float(request.form['Previous GPA']),
            'Past class failures': int(request.form['Past class failures']),
            'Study time per week': int(request.form['Study time per week']),
            'Attendance rate': float(request.form['Attendance rate']),
            'Parental education level': float(request.form['Parental education level']),
            'Socioeconomic status': request.form['Socioeconomic status'],
            'Motivation level': int(request.form['Motivation level']),
            'Self-discipline': int(request.form['Self-discipline']),
            'Health status': int(request.form['Health status']),
            'Hours of sleep': int(request.form['Hours of sleep']),
            'Parental involvement': int(request.form['Parental involvement']),
            'Peer support': int(request.form['Peer support']),
            'Participation in extracurricular activities': request.form['Participation in extracurricular activities']
        }
        # Load model and columns
        model = joblib.load('final_grade_xgb_model.pkl')
        # For one-hot encoding, match training columns
        df = pd.read_csv('dataset.csv')
        cat_cols = ['Gender', 'Socioeconomic status', 'Participation in extracurricular activities']
        X = df.drop(['Final grade (out of 100)'], axis=1)
        X_encoded = pd.get_dummies(X, columns=cat_cols)
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=cat_cols)
        input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        return redirect(url_for('result', pred=round(prediction, 2)))
    return render_template('predict.html')



# --- Feedback Route ---
@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form.get('helpful')
    comments = request.form.get('comments')
    pred = request.args.get('pred')
    # Save feedback to SQLite
    conn = sqlite3.connect('student_performance.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (feedback, comments) VALUES (?, ?)",
        (feedback, comments)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return render_template('result.html', prediction=pred, feedback_msg='Thank you for your feedback!')


# --- Chat Route ---
@app.route('/chat', methods=['POST'])
def chat():
    question = request.form.get('question')
    pred = request.args.get('pred')
    # Save chat question to SQLite
    conn = sqlite3.connect('student_performance.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_questions (question) VALUES (?)", (question,))
    conn.commit()
    cursor.close()
    conn.close()
    chat_response = f"You asked: {question}. (Live chat coming soon!)"
    return render_template('result.html', prediction=pred, chat_response=chat_response)

# --- Result Route ---
@app.route('/result', methods=['GET'])
def result():
    pred = request.args.get('pred', None)
    feedback_msg = request.args.get('feedback_msg')
    chat_response = request.args.get('chat_response')
    return render_template('result.html', prediction=pred, feedback_msg=feedback_msg, chat_response=chat_response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
