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
import math

# Import ML/DL libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Set page configuration
st.set_page_config(
    page_title="Rural EduGame Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
class CircuitBuilder:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.components = self.generate_components()
        self.correct_circuit = self.generate_correct_circuit()
        self.user_circuit = []
        self.score = 0
        
    def generate_components(self):
        return [
            {"id": "battery", "name": "Battery", "image": "🔋"},
            {"id": "bulb", "name": "Light Bulb", "image": "💡"},
            {"id": "switch", "name": "Switch", "image": "🔘"},
            {"id": "resistor", "name": "Resistor", "image": "📏"},
            {"id": "wire", "name": "Wire", "image": "➖"}
        ]
    
    def generate_correct_circuit(self):
        if self.grade_level <= 8:
            return ["battery", "wire", "switch", "wire", "bulb", "wire", "battery"]
        else:
            return ["battery", "wire", "resistor", "wire", "switch", "wire", "bulb", "wire", "battery"]
    
    def add_component(self, component_id):
        self.user_circuit.append(component_id)
        
    def check_circuit(self):
        return self.user_circuit == self.correct_circuit
    
    def get_score(self):
        if self.check_circuit():
            self.score = 100
        else:
            # Partial credit based on how many components are in correct position
            correct_positions = sum(1 for i, comp in enumerate(self.user_circuit) 
                                  if i < len(self.correct_circuit) and comp == self.correct_circuit[i])
            self.score = int((correct_positions / len(self.correct_circuit)) * 100)
        return self.score

class MathPuzzle:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.puzzle = self.generate_puzzle()
        self.solution = self.generate_solution()
        self.user_solution = np.zeros((3, 3)) if grade_level <= 8 else np.zeros((4, 4))
        self.score = 0
        
    def generate_puzzle(self):
        if self.grade_level <= 8:
            # Simple 3x3 magic square
            return np.array([[8, 0, 1], [0, 5, 0], [4, 0, 0]])
        else:
            # 4x4 magic square
            return np.array([[16, 0, 0, 13], [0, 11, 8, 0], [0, 7, 0, 0], [4, 0, 0, 1]])
    
    def generate_solution(self):
        if self.grade_level <= 8:
            # Magic constant is 15 for 3x3
            return np.array([[8, 3, 1], [6, 5, 4], [4, 7, 9]])
        else:
            # Magic constant is 34 for 4x4
            return np.array([[16, 3, 2, 13], [5, 11, 8, 10], [9, 7, 6, 12], [4, 15, 14, 1]])
    
    def update_cell(self, row, col, value):
        if self.puzzle[row, col] == 0:  # Only allow updates to empty cells
            self.user_solution[row, col] = value
    
    def check_solution(self):
        return np.array_equal(self.user_solution, self.solution)
    
    def get_score(self):
        if self.check_solution():
            self.score = 100
        else:
            # Calculate how many rows, columns and diagonals are correct
            correct_lines = 0
            total_lines = len(self.solution) * 2 + 2  # rows + columns + 2 diagonals
            
            # Check rows
            for i in range(len(self.solution)):
                if sum(self.user_solution[i, :]) == sum(self.solution[i, :]):
                    correct_lines += 1
            
            # Check columns
            for j in range(len(self.solution)):
                if sum(self.user_solution[:, j]) == sum(self.solution[:, j]):
                    correct_lines += 1
            
            # Check diagonals
            if sum(np.diag(self.user_solution)) == sum(np.diag(self.solution)):
                correct_lines += 1
            if sum(np.diag(np.fliplr(self.user_solution))) == sum(np.diag(np.fliplr(self.solution))):
                correct_lines += 1
            
            self.score = int((correct_lines / total_lines) * 100)
        return self.score

class EcosystemSimulator:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.organisms = self.generate_organisms()
        self.food_web = self.generate_food_web()
        self.user_web = {}
        self.score = 0
        
    def generate_organisms(self):
        if self.grade_level <= 8:
            return ["Grass", "Rabbit", "Fox", "Eagle"]
        else:
            return ["Phytoplankton", "Zooplankton", "Small Fish", "Large Fish", "Shark", "Decomposer"]
    
    def generate_food_web(self):
        if self.grade_level <= 8:
            return {
                "Grass": [],
                "Rabbit": ["Grass"],
                "Fox": ["Rabbit"],
                "Eagle": ["Rabbit", "Fox"]
            }
        else:
            return {
                "Phytoplankton": [],
                "Zooplankton": ["Phytoplankton"],
                "Small Fish": ["Zooplankton"],
                "Large Fish": ["Small Fish"],
                "Shark": ["Large Fish"],
                "Decomposer": ["Phytoplankton", "Zooplankton", "Small Fish", "Large Fish", "Shark"]
            }
    
    def add_relationship(self, predator, prey):
        if predator not in self.user_web:
            self.user_web[predator] = []
        if prey not in self.user_web[predator]:
            self.user_web[predator].append(prey)
    
    def check_web(self):
        return self.user_web == self.food_web
    
    def get_score(self):
        if self.check_web():
            self.score = 100
        else:
            # Calculate accuracy based on correct relationships
            correct_relationships = 0
            total_relationships = sum(len(prey) for prey in self.food_web.values())
            
            for predator, prey_list in self.food_web.items():
                if predator in self.user_web:
                    for prey in prey_list:
                        if prey in self.user_web[predator]:
                            correct_relationships += 1
            
            self.score = int((correct_relationships / total_relationships) * 100)
        return self.score

class LanguageAdventure:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.story = self.generate_story()
        self.missing_words = self.extract_missing_words()
        self.user_answers = {}
        self.score = 0
        
    def generate_story(self):
        if self.grade_level <= 8:
            return """
            Once upon a time, there was a [1] who lived in a small [2]. 
            Every day, he would [3] to the nearby forest to collect [4]. 
            One day, he discovered a [5] that changed his life forever.
            """
        else:
            return """
            In the realm of [1], where [2] theories prevailed, a remarkable [3] 
            was about to unfold. The [4] of the situation was not immediately 
            apparent to the [5], who continued their [6] research despite the 
            [7] circumstances that surrounded them.
            """
    
    def extract_missing_words(self):
        if self.grade_level <= 8:
            return {
                1: {"hint": "A person (noun)", "answer": "boy"},
                2: {"hint": "A type of home (noun)", "answer": "village"},
                3: {"hint": "An action (verb)", "answer": "go"},
                4: {"hint": "Something from nature (noun)", "answer": "berries"},
                5: {"hint": "Something valuable (noun)", "answer": "treasure"}
            }
        else:
            return {
                1: {"hint": "A field of study (noun)", "answer": "science"},
                2: {"hint": "A type of idea (adjective)", "answer": "complex"},
                3: {"hint": "An event (noun)", "answer": "discovery"},
                4: {"hint": "The nature of something (noun)", "answer": "gravity"},
                5: {"hint": "A person (noun)", "answer": "scientists"},
                6: {"hint": "An adjective describing work", "answer": "meticulous"},
                7: {"hint": "A challenging situation (adjective)", "answer": "difficult"}
            }
    
    def add_answer(self, blank_num, answer):
        self.user_answers[blank_num] = answer
    
    def check_answers(self):
        for num, data in self.missing_words.items():
            if num not in self.user_answers or self.user_answers[num].lower() != data["answer"].lower():
                return False
        return True
    
    def get_score(self):
        if self.check_answers():
            self.score = 100
        else:
            # Calculate score based on correct answers
            correct = 0
            total = len(self.missing_words)
            
            for num, data in self.missing_words.items():
                if num in self.user_answers and self.user_answers[num].lower() == data["answer"].lower():
                    correct += 1
            
            self.score = int((correct / total) * 100)
        return self.score

class GeographyExplorer:
    def __init__(self, grade_level):
        self.grade_level = grade_level
        self.countries = self.generate_countries()
        self.capitals = self.generate_capitals()
        self.user_matches = {}
        self.score = 0
        
    def generate_countries(self):
        if self.grade_level <= 8:
            return ["India", "United States", "Japan", "Brazil", "Egypt"]
        else:
            return ["India", "United States", "Japan", "Brazil", "Egypt", 
                   "Australia", "Germany", "South Africa", "China", "Mexico"]
    
    def generate_capitals(self):
        if self.grade_level <= 8:
            return {
                "India": "New Delhi",
                "United States": "Washington D.C.",
                "Japan": "Tokyo",
                "Brazil": "Brasília",
                "Egypt": "Cairo"
            }
        else:
            return {
                "India": "New Delhi",
                "United States": "Washington D.C.",
                "Japan": "Tokyo",
                "Brazil": "Brasília",
                "Egypt": "Cairo",
                "Australia": "Canberra",
                "Germany": "Berlin",
                "South Africa": "Pretoria",
                "China": "Beijing",
                "Mexico": "Mexico City"
            }
    
    def add_match(self, country, capital):
        self.user_matches[country] = capital
    
    def check_matches(self):
        return self.user_matches == self.capitals
    
    def get_score(self):
        if self.check_matches():
            self.score = 100
        else:
            # Calculate score based on correct matches
            correct = 0
            total = len(self.capitals)
            
            for country, capital in self.capitals.items():
                if country in self.user_matches and self.user_matches[country] == capital:
                    correct += 1
            
            self.score = int((correct / total) * 100)
        return self.score

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
            menu_options = ["Dashboard", "Circuit Builder", "Math Puzzle", "Ecosystem Simulator", 
                           "Language Adventure", "Geography Explorer", "My Progress"]
        else:
            menu_options = ["Dashboard", "Class Analytics", "Student Reports"]
        
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
                st.write("- Geography")
                st.write("- History")
            
            with col2:
                st.info("🎮 Learning Games")
                st.write("- Circuit Builder (Science)")
                st.write("- Math Puzzle (Mathematics)")
                st.write("- Ecosystem Simulator (Science)")
                st.write("- Language Adventure (Language Arts)")
                st.write("- Geography Explorer (Geography)")
            
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
                        st.write("- Try Circuit Builder to learn about electricity")
                        st.write("- Explore the Math Puzzle game")
                    else:
                        st.write("- Challenge yourself with the Ecosystem Simulator")
                        st.write("- Test your knowledge with Geography Explorer")
                
                with rec_col2:
                    st.write("**Popular among students:**")
                    st.write("- Circuit Builder - build working circuits")
                    st.write("- Language Adventure - creative storytelling")
        
        elif choice == "Circuit Builder":
            st.title("🔌 Circuit Builder")
            st.write("Build electrical circuits by dragging components to the workspace!")
            
            if st.session_state.current_game != "Circuit Builder":
                st.session_state.current_game = "Circuit Builder"
                st.session_state.game_state = CircuitBuilder(grade)
                st.rerun()
            
            game = st.session_state.game_state
            
            st.subheader("Available Components")
            cols = st.columns(len(game.components))
            for i, component in enumerate(game.components):
                with cols[i]:
                    st.write(f"{component['image']} {component['name']}")
                    if st.button(f"Add {component['name']}", key=f"add_{component['id']}"):
                        game.add_component(component['id'])
                        st.rerun()
            
            st.subheader("Your Circuit")
            if game.user_circuit:
                circuit_display = " → ".join([next(comp['image'] for comp in game.components if comp['id'] == c) for c in game.user_circuit])
                st.write(circuit_display)
            else:
                st.write("No components added yet. Start building your circuit!")
            
            if st.button("Check Circuit"):
                score = game.get_score()
                if score == 100:
                    st.success("🎉 Congratulations! Your circuit works perfectly!")
                else:
                    st.warning(f"Your circuit is {score}% correct. Try again!")
                
                save_game_progress(user_id, "Circuit Builder", "Science", score, grade, 10)
                
                if st.button("Play Again"):
                    st.session_state.game_state = CircuitBuilder(grade)
                    st.rerun()
        
        elif choice == "Math Puzzle":
            st.title("🧩 Math Puzzle")
            st.write("Solve the magic square by filling in the missing numbers!")
            
            if st.session_state.current_game != "Math Puzzle":
                st.session_state.current_game = "Math Puzzle"
                st.session_state.game_state = MathPuzzle(grade)
                st.rerun()
            
            game = st.session_state.game_state
            
            st.subheader("Puzzle Grid")
            grid_size = len(game.puzzle)
            for i in range(grid_size):
                cols = st.columns(grid_size)
                for j in range(grid_size):
                    with cols[j]:
                        if game.puzzle[i, j] != 0:
                            st.text_input("", value=str(game.puzzle[i, j]), key=f"fixed_{i}_{j}", disabled=True)
                        else:
                            value = int(game.user_solution[i, j]) if game.user_solution[i, j] != 0 else ""
                            new_value = st.number_input("", min_value=1, max_value=grid_size**2, 
                                                       value=value, key=f"input_{i}_{j}", 
                                                       format="%d")
                            if new_value != value:
                                game.update_cell(i, j, new_value)
                                st.rerun()
            
            if st.button("Check Solution"):
                score = game.get_score()
                if score == 100:
                    st.success("🎉 Congratulations! You solved the puzzle!")
                else:
                    st.warning(f"Your solution is {score}% correct. Keep trying!")
                
                save_game_progress(user_id, "Math Puzzle", "Mathematics", score, grade, 15)
                
                if st.button("Play Again"):
                    st.session_state.game_state = MathPuzzle(grade)
                    st.rerun()
        
        elif choice == "Ecosystem Simulator":
            st.title("🌿 Ecosystem Simulator")
            st.write("Create a food web by connecting organisms in their natural relationships!")
            
            if st.session_state.current_game != "Ecosystem Simulator":
                st.session_state.current_game = "Ecosystem Simulator"
                st.session_state.game_state = EcosystemSimulator(grade)
                st.rerun()
            
            game = st.session_state.game_state
            
            st.subheader("Organisms")
            for organism in game.organisms:
                st.write(f"- {organism}")
            
            st.subheader("Create Relationships")
            col1, col2 = st.columns(2)
            with col1:
                predator = st.selectbox("Predator (eats)", game.organisms)
            with col2:
                prey = st.selectbox("Prey (is eaten by)", game.organisms)
            
            if st.button("Add Relationship"):
                game.add_relationship(predator, prey)
                st.rerun()
            
            st.subheader("Your Food Web")
            if game.user_web:
                for predator, prey_list in game.user_web.items():
                    st.write(f"{predator} eats: {', '.join(prey_list)}")
            else:
                st.write("No relationships added yet.")
            
            if st.button("Check Food Web"):
                score = game.get_score()
                if score == 100:
                    st.success("🎉 Congratulations! Your food web is correct!")
                else:
                    st.warning(f"Your food web is {score}% correct. Keep trying!")
                
                save_game_progress(user_id, "Ecosystem Simulator", "Science", score, grade, 12)
                
                if st.button("Play Again"):
                    st.session_state.game_state = EcosystemSimulator(grade)
                    st.rerun()
        
        elif choice == "Language Adventure":
            st.title("📖 Language Adventure")
            st.write("Complete the story by filling in the missing words!")
            
            if st.session_state.current_game != "Language Adventure":
                st.session_state.current_game = "Language Adventure"
                st.session_state.game_state = LanguageAdventure(grade)
                st.rerun()
            
            game = st.session_state.game_state
            
            st.subheader("Story")
            # Display story with input boxes for missing words
            story_parts = game.story.split('[')
            display_text = story_parts[0]
            
            for i, part in enumerate(story_parts[1:]):
                num_end = part.find(']')
                blank_num = int(part[:num_end])
                rest_of_text = part[num_end+1:]
                
                hint = game.missing_words[blank_num]["hint"]
                current_value = game.user_answers.get(blank_num, "")
                
                display_text += f" [{blank_num}] " + rest_of_text
            
            st.write(display_text)
            
            st.subheader("Fill in the Blanks")
            for blank_num, data in game.missing_words.items():
                current_value = game.user_answers.get(blank_num, "")
                new_value = st.text_input(f"Word #{blank_num} ({data['hint']})", value=current_value, key=f"blank_{blank_num}")
                if new_value != current_value:
                    game.add_answer(blank_num, new_value)
            
            if st.button("Check Story"):
                score = game.get_score()
                if score == 100:
                    st.success("🎉 Congratulations! Your story is complete and correct!")
                    st.subheader("Completed Story")
                    completed_story = game.story
                    for blank_num, data in game.missing_words.items():
                        completed_story = completed_story.replace(f"[{blank_num}]", f"**{game.user_answers[blank_num]}**")
                    st.write(completed_story)
                else:
                    st.warning(f"Your story is {score}% correct. Keep trying!")
                
                save_game_progress(user_id, "Language Adventure", "Language Arts", score, grade, 15)
                
                if st.button("Play Again"):
                    st.session_state.game_state = LanguageAdventure(grade)
                    st.rerun()
        
        elif choice == "Geography Explorer":
            st.title("🌍 Geography Explorer")
            st.write("Match countries with their correct capitals!")
            
            if st.session_state.current_game != "Geography Explorer":
                st.session_state.current_game = "Geography Explorer"
                st.session_state.game_state = GeographyExplorer(grade)
                st.rerun()
            
            game = st.session_state.game_state
            
            st.subheader("Match Countries with Capitals")
            for country in game.countries:
                capitals_options = list(game.capitals.values()) + [""]
                current_capital = game.user_matches.get(country, "")
                new_capital = st.selectbox(f"Capital of {country}", options=capitals_options, 
                                         index=capitals_options.index(current_capital) if current_capital in capitals_options else 0,
                                         key=f"capital_{country}")
                if new_capital != current_capital:
                    game.add_match(country, new_capital)
            
            if st.button("Check Matches"):
                score = game.get_score()
                if score == 100:
                    st.success("🎉 Congratulations! All matches are correct!")
                else:
                    st.warning(f"Your matches are {score}% correct. Keep trying!")
                
                save_game_progress(user_id, "Geography Explorer", "Geography", score, grade, 10)
                
                if st.button("Play Again"):
                    st.session_state.game_state = GeographyExplorer(grade)
                    st.rerun()
        
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

if __name__ == "__main__":
    main()