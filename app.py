# main.py
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import time
import random
import os
from pathlib import Path
import hashlib
import base64

# Import ML/DL libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Rural EduGame Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini AI if available
if GEMINI_AVAILABLE:
    try:
        # You'll need to set up your API key
        # genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # gemini_model = genai.GenerativeModel('gemini-1.0-flash')
        AI_ENABLED = True
    except:
        AI_ENABLED = False
else:
    AI_ENABLED = False

# Database setup
def init_db():
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  user_type TEXT,
                  grade INTEGER,
                  school TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create game progress table
    c.execute('''CREATE TABLE IF NOT EXISTS game_progress
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  game_name TEXT,
                  subject TEXT,
                  score INTEGER,
                  level INTEGER,
                  time_spent INTEGER,
                  completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS analytics
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  date TEXT,
                  engagement_score INTEGER,
                  improvement_rate REAL,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create offline content table
    c.execute('''CREATE TABLE IF NOT EXISTS offline_content
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  content_name TEXT,
                  subject TEXT,
                  grade_level TEXT,
                  content_type TEXT,
                  content_data BLOB,
                  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# User authentication functions
def create_user(username, password, user_type, grade, school):
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, user_type, grade, school) VALUES (?, ?, ?, ?, ?)",
                  (username, hashed_password, user_type, grade, school))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def verify_user(username, password):
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user

# Game classes
class MathAdventure:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.questions = self.generate_questions()
        self.current_score = 0
        self.current_question = 0
        
    def generate_questions(self):
        # Different questions based on grade level
        if self.grade_level <= 8:
            return [
                {"question": "What is 15 + 27?", "options": ["42", "32", "52", "38"], "answer": "42"},
                {"question": "If a box contains 24 apples and you take away 7, how many are left?", "options": ["17", "16", "18", "15"], "answer": "17"},
                {"question": "What is 8 × 9?", "options": ["72", "81", "64", "76"], "answer": "72"},
                {"question": "What is ⅓ of 27?", "options": ["9", "6", "12", "15"], "answer": "9"},
                {"question": "Solve: 45 ÷ 5 = ?", "options": ["9", "8", "7", "10"], "answer": "9"}
            ]
        else:
            return [
                {"question": "Solve for x: 2x + 5 = 15", "options": ["5", "10", "7.5", "6"], "answer": "5"},
                {"question": "What is the value of π (pi) to two decimal places?", "options": ["3.14", "3.15", "3.12", "3.18"], "answer": "3.14"},
                {"question": "If a right triangle has sides of length 3 and 4, what is the length of the hypotenuse?", "options": ["5", "6", "7", "8"], "answer": "5"},
                {"question": "What is the square root of 144?", "options": ["12", "14", "16", "18"], "answer": "12"},
                {"question": "Simplify: (x² + 3x - 10) ÷ (x - 2)", "options": ["x + 5", "x - 5", "x + 2", "x - 2"], "answer": "x + 5"}
            ]
    
    def display_question(self):
        if self.current_question >= len(self.questions):
            return None
        
        q = self.questions[self.current_question]
        return q
    
    def check_answer(self, answer):
        if answer == self.questions[self.current_question]["answer"]:
            self.current_score += 10
            self.current_question += 1
            return True
        else:
            self.current_question += 1
            return False
    
    def get_score(self):
        return self.current_score

class ScienceExplorer:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.questions = self.generate_questions()
        self.current_score = 0
        self.current_question = 0
        
    def generate_questions(self):
        if self.grade_level <= 8:
            return [
                {"question": "Which planet is known as the Red Planet?", "options": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": "Mars"},
                {"question": "What is the process by which plants make their own food?", "options": ["Photosynthesis", "Respiration", "Digestion", "Transpiration"], "answer": "Photosynthesis"},
                {"question": "Which gas do plants absorb from the atmosphere?", "options": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"], "answer": "Carbon dioxide"},
                {"question": "What is the smallest unit of life?", "options": ["Cell", "Atom", "Molecule", "Tissue"], "answer": "Cell"},
                {"question": "Which of these is NOT a state of matter?", "options": ["Energy", "Solid", "Liquid", "Gas"], "answer": "Energy"}
            ]
        else:
            return [
                {"question": "What is the chemical symbol for gold?", "options": ["Au", "Ag", "Fe", "Go"], "answer": "Au"},
                {"question": "What is the speed of light in vacuum?", "options": ["299,792 km/s", "150,000 km/s", "450,000 km/s", "100,000 km/s"], "answer": "299,792 km/s"},
                {"question": "Which subatomic particle has a negative charge?", "options": ["Electron", "Proton", "Neutron", "Photon"], "answer": "Electron"},
                {"question": "What is the main gas found in Earth's atmosphere?", "options": ["Nitrogen", "Oxygen", "Carbon dioxide", "Hydrogen"], "answer": "Nitrogen"},
                {"question": "Which law states that energy cannot be created or destroyed?", "options": ["First Law of Thermodynamics", "Law of Conservation of Mass", "Newton's First Law", "Ohm's Law"], "answer": "First Law of Thermodynamics"}
            ]
    
    def display_question(self):
        if self.current_question >= len(self.questions):
            return None
        
        q = self.questions[self.current_question]
        return q
    
    def check_answer(self, answer):
        if answer == self.questions[self.current_question]["answer"]:
            self.current_score += 10
            self.current_question += 1
            return True
        else:
            self.current_question += 1
            return False
    
    def get_score(self):
        return self.current_score

class WordWizard:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.words = self.generate_words()
        self.current_score = 0
        self.current_word = 0
        
    def generate_words(self):
        if self.grade_level <= 8:
            return [
                {"word": "BENEvolent", "hint": "Kind and generous", "meaning": "Well-meaning and kindly"},
                {"word": "CAPitulate", "hint": "To surrender", "meaning": "Cease to resist an opponent or an unwelcome demand"},
                {"word": "DEference", "hint": "Humble submission and respect", "meaning": "Polite respect, especially putting another person's interests first"},
                {"word": "ENigma", "hint": "A puzzle or mystery", "meaning": "A person or thing that is mysterious or difficult to understand"},
                {"word": "FORTitude", "hint": "Courage in pain or adversity", "meaning": "Strength of mind that enables a person to encounter danger or bear pain with courage"}
            ]
        else:
            return [
                {"word": "ABNEGATION", "hint": "The act of renouncing or rejecting something", "meaning": "The act of rejecting or denying something"},
                {"word": "CACOPHONY", "hint": "A harsh, discordant mixture of sounds", "meaning": "A harsh, jarring sound"},
                {"word": "DEleterious", "hint": "Causing harm or damage", "meaning": "Causing harm or damage"},
                {"word": "EPHEmeral", "hint": "Lasting for a very short time", "meaning": "Lasting for a very short time"},
                {"word": "IDIOsyncrasy", "hint": "A mode of behavior peculiar to an individual", "meaning": "A characteristic or habit peculiar to an individual"}
            ]
    
    def display_word(self):
        if self.current_word >= len(self.words):
            return None
        
        word = self.words[self.current_word]
        scrambled = self.scramble_word(word["word"])
        return {"scrambled": scrambled, "hint": word["hint"], "actual": word["word"], "meaning": word["meaning"]}
    
    def scramble_word(self, word):
        letters = list(word)
        random.shuffle(letters)
        return ''.join(letters)
    
    def check_answer(self, answer):
        if answer.upper() == self.words[self.current_word]["word"].upper():
            self.current_score += 15
            self.current_word += 1
            return True
        else:
            self.current_word += 1
            return False
    
    def get_score(self):
        return self.current_score

class HistoryQuest:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.questions = self.generate_questions()
        self.current_score = 0
        self.current_question = 0
        
    def generate_questions(self):
        if self.grade_level <= 8:
            return [
                {"question": "Who was the first Prime Minister of India?", "options": ["Jawaharlal Nehru", "Mahatma Gandhi", "Sardar Patel", "Dr. Rajendra Prasad"], "answer": "Jawaharlal Nehru"},
                {"question": "In which year did India gain independence?", "options": ["1947", "1945", "1950", "1942"], "answer": "1947"},
                {"question": "Who was known as the 'Father of the Indian Constitution'?", "options": ["Dr. B.R. Ambedkar", "Mahatma Gandhi", "Jawaharlal Nehru", "Subhas Chandra Bose"], "answer": "Dr. B.R. Ambedkar"},
                {"question": "Which was the first metal used by humans?", "options": ["Copper", "Iron", "Bronze", "Gold"], "answer": "Copper"},
                {"question": "The Indus Valley Civilization was located in which modern-day country?", "options": ["Pakistan", "India", "Bangladesh", "Nepal"], "answer": "Pakistan"}
            ]
        else:
            return [
                {"question": "The Battle of Plassey was fought in which year?", "options": ["1757", "1764", "1748", "1772"], "answer": "1757"},
                {"question": "Who was the first woman Prime Minister of India?", "options": ["Indira Gandhi", "Sarojini Naidu", "Pratibha Patil", "Sonia Gandhi"], "answer": "Indira Gandhi"},
                {"question": "The ancient university of Nalanda was located in which Indian state?", "options": ["Bihar", "Uttar Pradesh", "Madhya Pradesh", "West Bengal"], "answer": "Bihar"},
                {"question": "Who wrote the book 'Discovery of India'?", "options": ["Jawaharlal Nehru", "Rabindranath Tagore", "Mahatma Gandhi", "Dr. Rajendra Prasad"], "answer": "Jawaharlal Nehru"},
                {"question": "The Quit India Movement was launched in which year?", "options": ["1942", "1930", "1947", "1920"], "answer": "1942"}
            ]
    
    def display_question(self):
        if self.current_question >= len(self.questions):
            return None
        
        q = self.questions[self.current_question]
        return q
    
    def check_answer(self, answer):
        if answer == self.questions[self.current_question]["answer"]:
            self.current_score += 10
            self.current_question += 1
            return True
        else:
            self.current_question += 1
            return False
    
    def get_score(self):
        return self.current_score

# AI-powered learning assistant
class LearningAssistant:
    def __init__(self):
        self.history = []
    
    def generate_explanation(self, topic, grade_level):
        if not AI_ENABLED:
            return "AI features are currently unavailable. Please check your API configuration."
        
        try:
            # In a real implementation, this would call the Gemini API
            explanations = {
                "photosynthesis": "Photosynthesis is the process plants use to convert sunlight into food. They take in carbon dioxide and water, and with the help of sunlight, create glucose (sugar) and release oxygen.",
                "algebra": "Algebra is like a puzzle where we use letters (called variables) to represent unknown numbers. The goal is to find what number the letter represents by solving equations.",
                "gravity": "Gravity is the force that pulls objects toward each other. On Earth, it's what makes things fall down and gives us weight. The larger the object, the stronger its gravitational pull.",
                "fractions": "Fractions represent parts of a whole. The top number (numerator) shows how many parts you have, and the bottom number (denominator) shows how many equal parts the whole is divided into."
            }
            
            if topic.lower() in explanations:
                return explanations[topic.lower()]
            else:
                return f"Here's a simple explanation of {topic} for a {grade_level}th grade student: {topic} is an important concept that helps us understand how things work in the world. You'll learn more about it as you continue your studies!"
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def generate_quiz(self, subject, grade_level, num_questions=5):
        if not AI_ENABLED:
            # Return a default quiz if AI is not available
            return [
                {"question": "What is the capital of France?", "options": ["Paris", "London", "Berlin", "Madrid"], "answer": "Paris"},
                {"question": "Which planet is known as the Red Planet?", "options": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": "Mars"},
                {"question": "What is 7 x 8?", "options": ["56", "64", "54", "72"], "answer": "56"},
                {"question": "Who wrote 'Romeo and Juliet'?", "options": ["William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain"], "answer": "William Shakespeare"},
                {"question": "What is the chemical symbol for water?", "options": ["H₂O", "CO₂", "O₂", "NaCl"], "answer": "H₂O"}
            ]
        
        try:
            # In a real implementation, this would call the Gemini API
            # For now, return a default quiz
            quizzes = {
                "math": [
                    {"question": "What is 15 + 27?", "options": ["42", "32", "52", "38"], "answer": "42"},
                    {"question": "If a box contains 24 apples and you take away 7, how many are left?", "options": ["17", "16", "18", "15"], "answer": "17"},
                    {"question": "What is 8 × 9?", "options": ["72", "81", "64", "76"], "answer": "72"},
                    {"question": "What is ⅓ of 27?", "options": ["9", "6", "12", "15"], "answer": "9"},
                    {"question": "Solve: 45 ÷ 5 = ?", "options": ["9", "8", "7", "10"], "answer": "9"}
                ],
                "science": [
                    {"question": "Which planet is known as the Red Planet?", "options": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": "Mars"},
                    {"question": "What is the process by which plants make their own food?", "options": ["Photosynthesis", "Respiration", "Digestion", "Transpiration"], "answer": "Photosynthesis"},
                    {"question": "Which gas do plants absorb from the atmosphere?", "options": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"], "answer": "Carbon dioxide"},
                    {"question": "What is the smallest unit of life?", "options": ["Cell", "Atom", "Molecule", "Tissue"], "answer": "Cell"},
                    {"question": "Which of these is NOT a state of matter?", "options": ["Energy", "Solid", "Liquid", "Gas"], "answer": "Energy"}
                ]
            }
            
            if subject.lower() in quizzes:
                return quizzes[subject.lower()]
            else:
                return quizzes["math"]  # Default to math quiz
        except Exception as e:
            return [{"question": f"Error generating quiz: {str(e)}", "options": ["Check", "API", "Configuration"], "answer": "Check"}]

# Analytics functions
def save_game_progress(user_id, game_name, subject, score, level, time_spent):
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    c.execute("INSERT INTO game_progress (user_id, game_name, subject, score, level, time_spent) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, game_name, subject, score, level, time_spent))
    conn.commit()
    conn.close()

def get_user_progress(user_id):
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    c.execute("SELECT game_name, subject, score, level, completed_at FROM game_progress WHERE user_id = ? ORDER BY completed_at DESC", (user_id,))
    progress = c.fetchall()
    conn.close()
    return progress

def get_class_progress(teacher_id):
    # This would typically get all students for a teacher
    # For simplicity, we'll get all progress
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    c.execute("""
        SELECT u.username, g.game_name, g.subject, AVG(g.score), COUNT(g.id)
        FROM game_progress g
        JOIN users u ON g.user_id = u.id
        WHERE u.user_type = 'student'
        GROUP BY u.username, g.game_name, g.subject
    """)
    progress = c.fetchall()
    conn.close()
    return progress

# ML functions for analytics
def analyze_student_performance(user_id):
    conn = sqlite3.connect('edu_game.db')
    df = pd.read_sql_query("SELECT game_name, subject, score, level, time_spent FROM game_progress WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    
    if df.empty:
        return "No data available for analysis"
    
    # Simple analysis
    avg_score = df['score'].mean()
    total_time = df['time_spent'].sum()
    favorite_subject = df['subject'].mode()[0] if not df['subject'].mode().empty else "None"
    
    # Use KMeans to cluster performance
    if len(df) > 5:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[['score', 'time_spent']])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        df['cluster'] = kmeans.labels_
        
        # Interpret clusters
        cluster_means = df.groupby('cluster')[['score', 'time_spent']].mean()
        if cluster_means.iloc[0]['score'] > cluster_means.iloc[1]['score']:
            high_perf_cluster = 0
        else:
            high_perf_cluster = 1
        
        improvement = "improving" if df[df['cluster'] == high_perf_cluster].index.max() > df[df['cluster'] != high_perf_cluster].index.max() else "needs improvement"
    else:
        improvement = "not enough data for trend analysis"
    
    analysis = f"""
    Performance Analysis:
    - Average Score: {avg_score:.2f}/100
    - Total Time Spent: {total_time} minutes
    - Favorite Subject: {favorite_subject}
    - Performance Trend: {improvement}
    
    Recommendations:
    - Focus on subjects where scores are lower
    - Try to maintain consistent study time
    - Revisit completed games to improve scores
    """
    
    return analysis

# Offline functionality
def save_content_for_offline(content_name, subject, grade_level, content_type, content_data):
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO offline_content (content_name, subject, grade_level, content_type, content_data) VALUES (?, ?, ?, ?, ?)",
              (content_name, subject, grade_level, content_type, content_data))
    conn.commit()
    conn.close()

def get_offline_content():
    conn = sqlite3.connect('edu_game.db')
    c = conn.cursor()
    c.execute("SELECT content_name, subject, grade_level, content_type, last_updated FROM offline_content")
    content = c.fetchall()
    conn.close()
    return content

# Main application
def main():
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_game' not in st.session_state:
        st.session_state.current_game = None
    if 'game_state' not in st.session_state:
        st.session_state.game_state = None
    
    # Sidebar for navigation
    st.sidebar.title("🎓 Rural EduGame Platform")
    
    if st.session_state.user is None:
        # Login/Signup section
        auth_option = st.sidebar.selectbox("Select Option", ["Login", "Sign Up"])
        
        if auth_option == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                user = verify_user(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
        
        else:  # Sign Up
            st.sidebar.subheader("Create Account")
            new_username = st.sidebar.text_input("Choose Username")
            new_password = st.sidebar.text_input("Choose Password", type="password")
            user_type = st.sidebar.selectbox("User Type", ["student", "teacher"])
            grade = st.sidebar.selectbox("Grade", range(6, 13)) if user_type == "student" else None
            school = st.sidebar.text_input("School Name")
            
            if st.sidebar.button("Create Account"):
                if create_user(new_username, new_password, user_type, grade, school):
                    st.sidebar.success("Account created successfully. Please login.")
                else:
                    st.sidebar.error("Username already exists")
    
    else:
        # User is logged in
        user_id, username, _, user_type, grade, school, _ = st.session_state.user
        
        st.sidebar.write(f"Welcome, {username}!")
        st.sidebar.write(f"Type: {user_type.capitalize()}")
        if user_type == "student":
            st.sidebar.write(f"Grade: {grade}")
        st.sidebar.write(f"School: {school}")
        
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.current_game = None
            st.session_state.game_state = None
            st.rerun()
        
        # Navigation based on user type
        if user_type == "student":
            menu_options = ["Dashboard", "Math Adventure", "Science Explorer", "Word Wizard", "History Quest", "AI Tutor", "My Progress", "Offline Content"]
        else:
            menu_options = ["Dashboard", "Class Analytics", "Student Reports", "Content Management"]
        
        choice = st.sidebar.selectbox("Menu", menu_options)
        
        # Main content area
        if choice == "Dashboard":
            st.title("🎓 Gamified Learning Platform for Rural Education")
            st.subheader("Welcome to your personalized learning dashboard!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("📚 Subjects Covered")
                st.write("- Mathematics")
                st.write("- Science")
                st.write("- Language Arts")
                st.write("- History/Social Studies")
            
            with col2:
                st.info("🎮 Learning Games")
                st.write("- Math Adventure")
                st.write("- Science Explorer")
                st.write("- Word Wizard")
                st.write("- History Quest")
            
            with col3:
                st.info("🏆 Your Progress")
                progress = get_user_progress(user_id)
                if progress:
                    st.write(f"Games Completed: {len(progress)}")
                    avg_score = sum(p[2] for p in progress) / len(progress)
                    st.write(f"Average Score: {avg_score:.1f}%")
                else:
                    st.write("No games completed yet")
            
            if user_type == "student":
                st.subheader("Recommended For You")
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.write("**Based on your grade level:**")
                    if grade <= 8:
                        st.write("- Practice basic algebra in Math Adventure")
                        st.write("- Learn about photosynthesis in Science Explorer")
                    else:
                        st.write("- Try advanced calculus problems")
                        st.write("- Explore quantum physics concepts")
                
                with rec_col2:
                    st.write("**Popular among students:**")
                    st.write("- Word Wizard vocabulary challenge")
                    st.write("- History Quest independence movement edition")
        
        elif choice == "Math Adventure":
            st.title("➗ Math Adventure")
            st.write("Embark on a journey through mathematical concepts and problems!")
            
            if st.session_state.current_game != "Math Adventure":
                st.session_state.current_game = "Math Adventure"
                st.session_state.game_state = MathAdventure(grade)
                st.rerun()
            
            game = st.session_state.game_state
            question = game.display_question()
            
            if question:
                st.subheader(f"Question {game.current_question + 1}")
                st.write(question["question"])
                
                answer = st.radio("Select your answer:", question["options"], key=f"math_q{game.current_question}")
                
                if st.button("Submit Answer"):
                    if game.check_answer(answer):
                        st.success("Correct! +10 points")
                    else:
                        st.error(f"Wrong! The correct answer is: {question['answer']}")
                    time.sleep(1)
                    st.rerun()
            else:
                st.success(f"Game completed! Your final score: {game.get_score()}")
                save_game_progress(user_id, "Math Adventure", "Mathematics", game.get_score(), grade, 10)
                if st.button("Play Again"):
                    st.session_state.game_state = MathAdventure(grade)
                    st.rerun()
        
        elif choice == "Science Explorer":
            st.title("🔬 Science Explorer")
            st.write("Discover the wonders of science through interactive exploration!")
            
            if st.session_state.current_game != "Science Explorer":
                st.session_state.current_game = "Science Explorer"
                st.session_state.game_state = ScienceExplorer(grade)
                st.rerun()
            
            game = st.session_state.game_state
            question = game.display_question()
            
            if question:
                st.subheader(f"Question {game.current_question + 1}")
                st.write(question["question"])
                
                answer = st.radio("Select your answer:", question["options"], key=f"science_q{game.current_question}")
                
                if st.button("Submit Answer"):
                    if game.check_answer(answer):
                        st.success("Correct! +10 points")
                    else:
                        st.error(f"Wrong! The correct answer is: {question['answer']}")
                    time.sleep(1)
                    st.rerun()
            else:
                st.success(f"Game completed! Your final score: {game.get_score()}")
                save_game_progress(user_id, "Science Explorer", "Science", game.get_score(), grade, 10)
                if st.button("Play Again"):
                    st.session_state.game_state = ScienceExplorer(grade)
                    st.rerun()
        
        elif choice == "Word Wizard":
            st.title("🔤 Word Wizard")
            st.write("Enhance your vocabulary and language skills with word challenges!")
            
            if st.session_state.current_game != "Word Wizard":
                st.session_state.current_game = "Word Wizard"
                st.session_state.game_state = WordWizard(grade)
                st.rerun()
            
            game = st.session_state.game_state
            word_data = game.display_word()
            
            if word_data:
                st.subheader(f"Word {game.current_word + 1}")
                st.write(f"Unscramble this word: **{word_data['scrambled']}**")
                st.write(f"Hint: {word_data['hint']}")
                
                answer = st.text_input("Your answer:", key=f"word_q{game.current_word}")
                
                if st.button("Submit Answer"):
                    if game.check_answer(answer):
                        st.success("Correct! +15 points")
                        st.write(f"Meaning: {word_data['meaning']}")
                    else:
                        st.error(f"Wrong! The correct word is: {word_data['actual']}")
                        st.write(f"Meaning: {word_data['meaning']}")
                    time.sleep(2)
                    st.rerun()
            else:
                st.success(f"Game completed! Your final score: {game.get_score()}")
                save_game_progress(user_id, "Word Wizard", "Language Arts", game.get_score(), grade, 10)
                if st.button("Play Again"):
                    st.session_state.game_state = WordWizard(grade)
                    st.rerun()
        
        elif choice == "History Quest":
            st.title("🏛️ History Quest")
            st.write("Travel through time and explore historical events and figures!")
            
            if st.session_state.current_game != "History Quest":
                st.session_state.current_game = "History Quest"
                st.session_state.game_state = HistoryQuest(grade)
                st.rerun()
            
            game = st.session_state.game_state
            question = game.display_question()
            
            if question:
                st.subheader(f"Question {game.current_question + 1}")
                st.write(question["question"])
                
                answer = st.radio("Select your answer:", question["options"], key=f"history_q{game.current_question}")
                
                if st.button("Submit Answer"):
                    if game.check_answer(answer):
                        st.success("Correct! +10 points")
                    else:
                        st.error(f"Wrong! The correct answer is: {question['answer']}")
                    time.sleep(1)
                    st.rerun()
            else:
                st.success(f"Game completed! Your final score: {game.get_score()}")
                save_game_progress(user_id, "History Quest", "History", game.get_score(), grade, 10)
                if st.button("Play Again"):
                    st.session_state.game_state = HistoryQuest(grade)
                    st.rerun()
        
        elif choice == "AI Tutor":
            st.title("🤖 AI-Powered Learning Assistant")
            st.write("Get personalized explanations and help on any topic!")
            
            topic = st.text_input("Enter a topic you want to learn about:")
            if st.button("Explain This Topic"):
                if topic:
                    with st.spinner("Generating explanation..."):
                        assistant = LearningAssistant()
                        explanation = assistant.generate_explanation(topic, grade)
                        st.write(explanation)
                        
                        # Option to save for offline use
                        if st.button("Save for Offline Use"):
                            save_content_for_offline(
                                f"Explanation: {topic}",
                                "General",
                                grade,
                                "text",
                                explanation.encode()
                            )
                            st.success("Content saved for offline use!")
                else:
                    st.warning("Please enter a topic")
            
            st.subheader("Generate Practice Quiz")
            subject = st.selectbox("Select Subject", ["Math", "Science", "English", "History"])
            if st.button("Generate Quiz"):
                with st.spinner("Creating quiz questions..."):
                    assistant = LearningAssistant()
                    quiz = assistant.generate_quiz(subject, grade)
                    
                    for i, q in enumerate(quiz):
                        st.write(f"**Q{i+1}: {q['question']}**")
                        for opt in q['options']:
                            st.write(f"- {opt}")
                        st.write(f"**Answer:** {q['answer']}")
                        st.write("---")
        
        elif choice == "My Progress":
            st.title("📊 My Learning Progress")
            progress = get_user_progress(user_id)
            
            if progress:
                st.subheader("Game Performance")
                df = pd.DataFrame(progress, columns=["Game", "Subject", "Score", "Level", "Date"])
                
                # Show recent activity
                st.dataframe(df.head(10))
                
                # Show charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Scores by Subject")
                    subject_scores = df.groupby("Subject")["Score"].mean()
                    st.bar_chart(subject_scores)
                
                with col2:
                    st.subheader("Progress Over Time")
                    df['Date'] = pd.to_datetime(df['Date'])
                    time_series = df.set_index('Date')['Score']
                    st.line_chart(time_series)
                
                # Performance analysis
                st.subheader("Performance Analysis")
                analysis = analyze_student_performance(user_id)
                st.write(analysis)
            else:
                st.info("You haven't completed any games yet. Play some games to see your progress here!")
        
        elif choice == "Offline Content":
            st.title("📥 Offline Content")
            st.write("Access learning materials without an internet connection")
            
            content = get_offline_content()
            if content:
                for c in content:
                    st.write(f"**{c[0]}** ({c[1]}) - Grade {c[2]} - Last updated: {c[4]}")
            else:
                st.info("No offline content available yet. Use the AI Tutor to generate and save content.")
            
            st.subheader("Download Games for Offline Use")
            st.warning("Offline game functionality requires additional implementation for full functionality")
        
        elif choice == "Class Analytics" and user_type == "teacher":
            st.title("👨‍🏫 Class Analytics")
            progress = get_class_progress(user_id)
            
            if progress:
                df = pd.DataFrame(progress, columns=["Student", "Game", "Subject", "Avg Score", "Games Played"])
                
                st.subheader("Overall Class Performance")
                st.dataframe(df)
                
                st.subheader("Average Scores by Subject")
                subject_avg = df.groupby("Subject")["Avg Score"].mean()
                st.bar_chart(subject_avg)
                
                st.subheader("Student Engagement")
                engagement = df.groupby("Student")["Games Played"].sum()
                st.bar_chart(engagement)
            else:
                st.info("No student data available yet.")
        
        elif choice == "Student Reports" and user_type == "teacher":
            st.title("📝 Individual Student Reports")
            
            # Get list of students
            conn = sqlite3.connect('edu_game.db')
            c = conn.cursor()
            c.execute("SELECT id, username, grade FROM users WHERE user_type = 'student'")
            students = c.fetchall()
            conn.close()
            
            if students:
                student_options = [f"{s[1]} (Grade {s[2]})" for s in students]
                selected_student = st.selectbox("Select Student", student_options)
                
                if selected_student:
                    student_id = students[student_options.index(selected_student)][0]
                    progress = get_user_progress(student_id)
                    
                    if progress:
                        st.subheader(f"Progress Report for {selected_student}")
                        df = pd.DataFrame(progress, columns=["Game", "Subject", "Score", "Level", "Date"])
                        st.dataframe(df)
                        
                        # Show analysis
                        analysis = analyze_student_performance(student_id)
                        st.write(analysis)
                    else:
                        st.info("This student hasn't completed any games yet.")
            else:
                st.info("No students registered yet.")
        
        elif choice == "Content Management" and user_type == "teacher":
            st.title("📋 Content Management")
            st.write("Create and manage learning content for your students")
            
            st.subheader("Add New Content")
            content_name = st.text_input("Content Title")
            content_subject = st.selectbox("Subject", ["Math", "Science", "English", "History", "General"])
            content_grade = st.selectbox("Grade Level", range(6, 13))
            content_type = st.selectbox("Content Type", ["Lesson", "Worksheet", "Quiz", "Reference"])
            content_data = st.text_area("Content (for text) or upload instructions")
            
            if st.button("Save Content"):
                save_content_for_offline(
                    content_name,
                    content_subject,
                    content_grade,
                    content_type,
                    content_data.encode()
                )
                st.success("Content saved successfully!")
            
            st.subheader("Existing Content")
            content = get_offline_content()
            if content:
                for c in content:
                    st.write(f"**{c[0]}** ({c[1]}) - Grade {c[2]} - Type: {c[3]} - Last updated: {c[4]}")
            else:
                st.info("No content available yet.")

if __name__ == "__main__":
    main()