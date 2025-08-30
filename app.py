# Enhanced Resume Analyzer - Complete Application
"""
Advanced Resume Analyzer with:
- Sidebar Login/Signup/Logout
- Advanced AI & NLP Analysis
- Pie Charts & Gauge Visualizations
- PDF Report Generation
- User Profile & Dashboard
- Missing Skills & Enhancement Analysis
"""

import streamlit as st
import os
import io
import sqlite3
import hashlib
import datetime
import base64
import re
import json
import uuid
from collections import Counter
import tempfile

# Document processing
import PyPDF2
import docx

# Data analysis and visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Wedge, Circle

# AI/NLP libraries
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# PDF generation
from fpdf import FPDF
from PIL import Image

# Authentication
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "resume_analyzer.db")
UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
REPORTS_DIR = os.path.join(APP_DIR, "reports")
PROFILES_DIR = os.path.join(APP_DIR, "profiles")

# Create directories
for dir_path in [UPLOADS_DIR, REPORTS_DIR, PROFILES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# DATABASE AND AUTHENTICATION
# =============================================================================

def init_database():
    """Initialize SQLite database with all necessary tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            bio TEXT,
            avatar_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    # Resume analyses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            candidate_name TEXT,
            job_title TEXT,
            resume_text TEXT,
            job_description TEXT,
            match_score REAL,
            matching_skills TEXT,
            missing_skills TEXT,
            enhancement_suggestions TEXT,
            analysis_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # User sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash password with bcrypt or fallback to SHA256"""
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    else:
        # Fallback to salted SHA256
        salt = "resume_analyzer_salt_2024"
        return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    if BCRYPT_AVAILABLE and hashed.startswith('$2b$'):
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    else:
        # Fallback verification
        salt = "resume_analyzer_salt_2024"
        return hashed == hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def create_user(username: str, email: str, password: str, full_name: str = "") -> tuple:
    """Create new user account"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        hashed_pw = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
        """, (username, email, hashed_pw, full_name))
        
        conn.commit()
        return True, "Account created successfully!"
    
    except sqlite3.IntegrityError as e:
        if "username" in str(e).lower():
            return False, "Username already exists"
        elif "email" in str(e).lower():
            return False, "Email already exists"
        else:
            return False, "Account creation failed"
    
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user and return user info"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, password_hash, full_name, bio, avatar_path
        FROM users WHERE username = ? OR email = ?
    """, (username, username))
    
    user = cursor.fetchone()
    
    if user and verify_password(password, user[3]):
        # Update last login
        cursor.execute("""
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        """, (user[0],))
        conn.commit()
        
        conn.close()
        return {
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "full_name": user[4] or "",
            "bio": user[5] or "",
            "avatar_path": user[6]
        }
    
    conn.close()
    return None

def get_user_by_id(user_id: int) -> dict:
    """Get user information by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, full_name, bio, avatar_path, created_at, last_login
        FROM users WHERE id = ?
    """, (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "full_name": user[3] or "",
            "bio": user[4] or "",
            "avatar_path": user[5],
            "created_at": user[6],
            "last_login": user[7]
        }
    return None

def update_user_profile(user_id: int, full_name: str = None, bio: str = None, avatar_path: str = None) -> bool:
    """Update user profile information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if full_name is not None:
        updates.append("full_name = ?")
        params.append(full_name)
    
    if bio is not None:
        updates.append("bio = ?")
        params.append(bio)
    
    if avatar_path is not None:
        updates.append("avatar_path = ?")
        params.append(avatar_path)
    
    if updates:
        params.append(user_id)
        cursor.execute(f"""
            UPDATE users SET {', '.join(updates)} WHERE id = ?
        """, params)
        conn.commit()
    
    conn.close()
    return True

# =============================================================================
# AI AND NLP PROCESSING
# =============================================================================

# Comprehensive skills database
TECH_SKILLS = {
    'programming_languages': [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
        'Ruby', 'PHP', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'SQL'
    ],
    'web_technologies': [
        'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js',
        'Django', 'Flask', 'FastAPI', 'Spring Boot', 'ASP.NET', 'Laravel'
    ],
    'databases': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle',
        'Cassandra', 'Neo4j', 'DynamoDB', 'Elasticsearch'
    ],
    'cloud_platforms': [
        'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Heroku',
        'Digital Ocean', 'Terraform', 'CloudFormation'
    ],
    'data_science': [
        'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
        'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Keras',
        'Computer Vision', 'NLP', 'Statistics', 'Tableau', 'Power BI'
    ],
    'tools_frameworks': [
        'Git', 'GitHub', 'GitLab', 'Jenkins', 'CI/CD', 'Agile', 'Scrum',
        'JIRA', 'Confluence', 'Linux', 'Unix', 'Bash', 'PowerShell'
    ]
}

# Flatten all skills into a single list
ALL_SKILLS = []
for category in TECH_SKILLS.values():
    ALL_SKILLS.extend(category)

# Stopwords for filtering
STOPWORDS = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'the', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 'each', 'which',
    'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
    'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
    'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made',
    'may', 'part'
])

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model with caching"""
    if SBERT_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"Could not load Sentence Transformer: {e}")
            return None
    return None

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF, DOCX, or TXT)"""
    try:
        if uploaded_file.type == "application/pdf":
            # PDF extraction
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        
        elif uploaded_file.type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            # DOCX extraction
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            # TXT file
            text = str(uploaded_file.read(), "utf-8")
            return text
            
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def clean_and_tokenize(text):
    """Clean text and extract meaningful tokens"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s+#.-]', ' ', text.lower())
    
    # Split into words
    words = text.split()
    
    # Filter out stopwords and short words
    filtered_words = [
        word for word in words 
        if word not in STOPWORDS and len(word) > 2
    ]
    
    return filtered_words

def extract_skills_from_text(text):
    """Extract skills from text using pattern matching"""
    text_lower = text.lower()
    found_skills = []
    
    # Check for each skill in our database
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            # Verify it's a whole word match
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
    
    # Also extract potential skills using NER-like patterns
    # Look for capitalized words that might be technologies
    tech_pattern = r'\b[A-Z][a-zA-Z]*(?:[\s+.-][A-Z][a-zA-Z]*)*\b'
    potential_skills = re.findall(tech_pattern, text)
    
    for skill in potential_skills:
        if len(skill) > 2 and skill not in found_skills:
            found_skills.append(skill)
    
    return list(set(found_skills))

def calculate_semantic_similarity(text1, text2, model=None):
    """Calculate semantic similarity between two texts"""
    if model and SBERT_AVAILABLE:
        try:
            # Use Sentence Transformer for semantic similarity
            embeddings = model.encode([text1, text2])
            similarity = st_util.cos_sim(embeddings[0], embeddings[1]).item()
            return similarity
        except Exception:
            pass
    
    # Fallback to TF-IDF similarity
    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer().fit([text1, text2])
            vectors = vectorizer.transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except Exception:
            pass
    
    # Simple word overlap fallback
    words1 = set(clean_and_tokenize(text1))
    words2 = set(clean_and_tokenize(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def advanced_resume_analysis(resume_text, job_description):
    """Perform advanced analysis of resume against job description"""
    # Load model
    model = load_sentence_transformer()
    
    # Extract skills from both texts
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_description)
    
    # Find matching skills
    matching_skills = list(set(resume_skills).intersection(set(job_skills)))
    
    # Find missing skills
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    
    # Calculate overall similarity
    overall_similarity = calculate_semantic_similarity(resume_text, job_description, model)
    
    # Calculate match score (0-100)
    if len(job_skills) > 0:
        skill_match_ratio = len(matching_skills) / len(job_skills)
        match_score = (skill_match_ratio * 0.6 + overall_similarity * 0.4) * 100
    else:
        match_score = overall_similarity * 100
    
    # Generate enhancement suggestions
    enhancement_suggestions = generate_enhancement_suggestions(
        missing_skills, resume_text, job_description
    )
    
    return {
        'match_score': round(match_score, 2),
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'all_resume_skills': resume_skills,
        'all_job_skills': job_skills,
        'enhancement_suggestions': enhancement_suggestions,
        'overall_similarity': round(overall_similarity, 3),
        'analysis_method': 'Advanced NLP' if model else 'Standard Analysis'
    }

def generate_enhancement_suggestions(missing_skills, resume_text, job_description):
    """Generate specific enhancement suggestions based on missing skills"""
    suggestions = []
    
    for skill in missing_skills[:10]:  # Limit to top 10 missing skills
        if skill in TECH_SKILLS['programming_languages']:
            suggestions.append({
                'skill': skill,
                'category': 'Programming Language',
                'suggestion': f"Consider adding {skill} projects to your portfolio. Include specific examples of applications or systems you've built.",
                'priority': 'High' if 'senior' in job_description.lower() else 'Medium'
            })
        
        elif skill in TECH_SKILLS['web_technologies']:
            suggestions.append({
                'skill': skill,
                'category': 'Web Technology',
                'suggestion': f"Develop web applications using {skill}. Showcase responsive design and modern development practices.",
                'priority': 'High' if any(web in job_description.lower() for web in ['frontend', 'backend', 'fullstack']) else 'Medium'
            })
        
        elif skill in TECH_SKILLS['cloud_platforms']:
            suggestions.append({
                'skill': skill,
                'category': 'Cloud Platform',
                'suggestion': f"Gain hands-on experience with {skill}. Consider getting certified and deploying projects on the platform.",
                'priority': 'High' if 'cloud' in job_description.lower() else 'Medium'
            })
        
        elif skill in TECH_SKILLS['data_science']:
            suggestions.append({
                'skill': skill,
                'category': 'Data Science',
                'suggestion': f"Build data science projects using {skill}. Include data visualization and statistical analysis examples.",
                'priority': 'High' if any(ds in job_description.lower() for ds in ['data', 'analytics', 'ml', 'ai']) else 'Medium'
            })
        
        else:
            suggestions.append({
                'skill': skill,
                'category': 'Technical Skill',
                'suggestion': f"Enhance your knowledge in {skill}. Consider online courses and practical projects.",
                'priority': 'Medium'
            })
    
    return suggestions

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_match_score_gauge(score):
    """Create an interactive gauge chart for match score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score (%)", 'font': {'size': 20}},
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        width=400,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_skills_pie_chart(matching_skills, missing_skills):
    """Create a pie chart showing matching vs missing skills"""
    labels = ['Matching Skills', 'Missing Skills']
    values = [len(matching_skills), len(missing_skills)]
    colors = ['#2E8B57', '#DC143C']  # Sea Green and Crimson
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # Donut chart
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={'text': 'Skills Analysis Overview', 'x': 0.5, 'font': {'size': 18}},
        width=400,
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_skills_category_chart(skills_analysis):
    """Create a bar chart showing skills by category"""
    categories = []
    matching_counts = []
    missing_counts = []
    
    # Count skills by category
    for category, skills in TECH_SKILLS.items():
        category_name = category.replace('_', ' ').title()
        
        matching_in_cat = sum(1 for skill in skills_analysis['matching_skills'] 
                             if skill in skills)
        missing_in_cat = sum(1 for skill in skills_analysis['missing_skills'] 
                            if skill in skills)
        
        if matching_in_cat > 0 or missing_in_cat > 0:
            categories.append(category_name)
            matching_counts.append(matching_in_cat)
            missing_counts.append(missing_in_cat)
    
    fig = go.Figure(data=[
        go.Bar(name='Matching', x=categories, y=matching_counts, marker_color='#2E8B57'),
        go.Bar(name='Missing', x=categories, y=missing_counts, marker_color='#DC143C')
    ])
    
    fig.update_layout(
        barmode='group',
        title={'text': 'Skills Analysis by Category', 'x': 0.5, 'font': {'size': 18}},
        xaxis_title="Skill Categories",
        yaxis_title="Number of Skills",
        width=800,
        height=400,
        margin=dict(l=50, r=50, t=80, b=100),
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_skills_heatmap(resume_skills, job_skills):
    """Create a heatmap showing skill overlap"""
    # Create a simple skills matrix
    all_unique_skills = list(set(resume_skills + job_skills))
    
    if len(all_unique_skills) > 20:
        all_unique_skills = all_unique_skills[:20]  # Limit for readability
    
    matrix = []
    labels = []
    
    for skill in all_unique_skills:
        row = []
        if skill in resume_skills and skill in job_skills:
            row = [2]  # Both have it
            labels.append(f"{skill} (Match)")
        elif skill in resume_skills:
            row = [1]  # Only resume has it
            labels.append(f"{skill} (Resume only)")
        elif skill in job_skills:
            row = [0]  # Only job has it
            labels.append(f"{skill} (Job only)")
        else:
            continue
        
        matrix.append(row)
    
    if matrix:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            y=labels,
            x=['Skills'],
            colorscale=[[0, '#DC143C'], [0.5, '#FFD700'], [1, '#2E8B57']],
            showscale=True,
            colorbar=dict(
                tickvals=[0, 1, 2],
                ticktext=['Job Only', 'Resume Only', 'Match']
            )
        ))
        
        fig.update_layout(
            title={'text': 'Skills Overlap Analysis', 'x': 0.5, 'font': {'size': 18}},
            width=600,
            height=max(400, len(labels) * 25),
            margin=dict(l=200, r=50, t=80, b=50)
        )
        
        return fig
    
    return None

def create_enhancement_priority_chart(enhancement_suggestions):
    """Create a chart showing enhancement priorities"""
    if not enhancement_suggestions:
        return None
    
    priorities = {'High': 0, 'Medium': 0, 'Low': 0}
    skills_by_priority = {'High': [], 'Medium': [], 'Low': []}
    
    for suggestion in enhancement_suggestions:
        priority = suggestion.get('priority', 'Medium')
        priorities[priority] += 1
        skills_by_priority[priority].append(suggestion['skill'])
    
    # Create a funnel chart for priorities
    fig = go.Figure(go.Funnel(
        y=["High Priority", "Medium Priority", "Low Priority"],
        x=[priorities['High'], priorities['Medium'], priorities['Low']],
        textinfo="value+percent initial",
        marker=dict(color=["red", "orange", "yellow"])
    ))
    
    fig.update_layout(
        title={'text': 'Enhancement Priorities', 'x': 0.5, 'font': {'size': 18}},
        width=500,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# -----------------------
# Advanced analyzer (improved matching / missing / enhancement)
# -----------------------
def analyze_resume_advanced(resume_text, job_desc, top_k_sentences=6):
    resume_text_clean = clean_text(resume_text)
    jd_clean = clean_text(job_desc)

    job_sentences = [s.strip() for s in re.split(r'(?<=[\.\n;])\s+', jd_clean) if s.strip()]
    resume_sentences = [s.strip() for s in re.split(r'(?<=[\.\n;])\s+', resume_text_clean) if s.strip()]

    method = "overlap"
    if S_BERT_AVAILABLE and get_sentence_model() is not None:
        method = "sentence-transformer"
    elif SKLEARN_AVAILABLE:
        method = "tfidf"

    sims = semantic_similarity_matrix(job_sentences, resume_sentences) if job_sentences and resume_sentences else (np.zeros((len(job_sentences), max(1, len(resume_sentences)))) if NUMPY_AVAILABLE else [])

    job_best = []
    for i, js in enumerate(job_sentences):
        if resume_sentences:
            best_idx = int(sims[i].argmax()) if NUMPY_AVAILABLE else 0
            best_score = float(float(np.max(sims[i])) if NUMPY_AVAILABLE else 0.0)
            matched_resume = resume_sentences[best_idx] if best_idx >= 0 and resume_sentences else ""
        else:
            best_idx = -1
            best_score = 0.0
            matched_resume = ""
        job_best.append({"job_sentence": js, "best_resume_sentence": matched_resume, "score": best_score})

    job_best_sorted = sorted(job_best, key=lambda x: x["score"], reverse=True)
    top_sentences = job_best_sorted[:top_k_sentences]

    # tokens and frequency
    resume_tokens = tokenize(resume_text_clean)
    jd_tokens = tokenize(jd_clean)
    r_counter = Counter(resume_tokens)
    j_counter = Counter(jd_tokens)

    # more reliable candidate lists (filter stopwords and short tokens)
    resume_skill_candidates = [tok for tok,_ in r_counter.most_common(400) if tok in COMMON_SKILLS or len(tok) > 2]
    jd_skill_candidates = [tok for tok,_ in j_counter.most_common(500) if tok in COMMON_SKILLS or len(tok) > 2]

    matches = []
    for jd_skill in jd_skill_candidates:
        best = ("", 0.0)
        # try match against resume tokens
        for rs in resume_skill_candidates:
            score = 0.0
            if S_BERT_AVAILABLE and get_sentence_model() is not None:
                try:
                    model = get_sentence_model()
                    a = model.encode([jd_skill], convert_to_tensor=True)
                    b = model.encode([rs], convert_to_tensor=True)
                    sc = float(st_util.cos_sim(a, b).cpu().numpy()[0][0])
                    score = sc
                except Exception:
                    score = fuzzy_score(jd_skill, rs)
            elif SKLEARN_AVAILABLE:
                try:
                    vect = TfidfVectorizer().fit([jd_skill, rs])
                    A = vect.transform([jd_skill])
                    B = vect.transform([rs])
                    score = float(cosine_similarity(A, B)[0][0])
                except Exception:
                    score = fuzzy_score(jd_skill, rs)
            else:
                score = fuzzy_score(jd_skill, rs)
            if score > best[1]:
                best = (rs, score)
        matches.append({"jd_skill": jd_skill, "best_match_in_resume": best[0], "score": best[1]})

    # adaptive threshold
    MATCH_THRESHOLD = 0.45 if method == "sentence-transformer" else 0.55 if method == "tfidf" else 0.7
    matching = [m["jd_skill"] for m in matches if m["score"] >= MATCH_THRESHOLD]
    missing = [m["jd_skill"] for m in matches if m["score"] < MATCH_THRESHOLD]

    # importance scoring from job sentences
    jd_importance = Counter()
    for jb in job_best:
        for tk in tokenize(jb["job_sentence"]):
            jd_importance[tk] += jb["score"] + 0.01

    # suggestions prioritized
    suggestions = []
    for s in missing:
        importance = jd_importance[s] if s in jd_importance else 0.0
        suggestions.append({"skill": s, "importance": importance, "suggestion": f"Add concrete experience demonstrating '{s}' (project, duration, result)"} )
    suggestions_sorted = sorted(suggestions, key=lambda x: x["importance"], reverse=True)
    suggestions_text = [s["suggestion"] for s in suggestions_sorted]

    # compute match score (weighted)
    total_weight = 0.0
    covered_weight = 0.0
    for jb in job_best:
        sent = jb["job_sentence"]
        weight = sum(jd_importance.get(tok,0.0) for tok in tokenize(sent)) + 0.5
        total_weight += weight
        sent_tokens = tokenize(sent)
        if any(tok in matching for tok in sent_tokens):
            covered_weight += weight * min(1.0, jb["score"] + 0.2)
    match_score = (covered_weight / total_weight) * 100.0 if total_weight > 0 else (len(matching) / max(1, len(matching) + len(missing)) * 100.0)

    # build breakdown table
    breakdown = []
    for m in matches:
        status = "Matched" if m["score"] >= MATCH_THRESHOLD else "Missing"
        breakdown.append({"jd_token": m["jd_skill"], "best_match": m["best_match_in_resume"], "score": float(m["score"]), "status": status})

    # enhancement bullets for missing skills (short generative templates)
    def bullets_for(skill):
        return [
            f"Built a {skill}-based project delivering measurable results (e.g., decreased latency by X%).",
            f"Implemented {skill} in a team setting and documented architecture, tests, and deployment pipeline."
        ]
    enhancements = {s["skill"]: {"priority": s["importance"], "bullets": bullets_for(s["skill"]), "resources": []} for s in suggestions_sorted[:50]}

    return {
        "method_used": method,
        "matching_skills": matching,
        "missing_skills": missing,
        "suggestions": suggestions_text[:40],
        "match_score": match_score,
        "top_sentences": top_sentences,
        "breakdown": breakdown,
        "enhancements": enhancements
    }

# -----------------------
# Visuals: gauge (plotly) + matplotlib fallback
# -----------------------
def plotly_gauge(match_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_score,
        number={'suffix': "%"},
        title={'text': "Match Score"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range':[0,40], 'color':'rgba(255,0,0,0.5)'},
                {'range':[40,70], 'color':'rgba(255,165,0,0.5)'},
                {'range':[70,100], 'color':'rgba(0,128,0,0.5)'}
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10))
    return fig

def matplotlib_gauge(match_score, figsize=(4,2.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-0.1,1.2)
    ax.axis('off')
    wedges = [
        (0, 0.4, "red"),
        (0.4, 0.7, "orange"),
        (0.7, 1.0, "green")
    ]
    start = 180
    for a,b,color in wedges:
        theta1 = start - a*180
        theta2 = start - b*180
        wedge = Wedge((0,0), 1.0, theta2, theta1, facecolor=color, alpha=0.6)
        ax.add_patch(wedge)
    frac = max(0.0, min(1.0, match_score/100.0))
    angle = math.radians(180 - frac*180)
    x = math.cos(angle)
    y = math.sin(angle)
    ax.plot([0, x*0.9], [0, y*0.9], lw=3, color="black")
    circ = Circle((0,0), 0.06, color="black")
    ax.add_patch(circ)
    ax.text(0, -0.05, f"{match_score:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# -----------------------
# PDF report generation (working)
# -----------------------
def generate_pdf_report(candidate_name, job_title, match_score, matching, missing, suggestions, top_sentences, method_used, pie_fig=None):
    # Create report bytes and return
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Resume Analysis Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 85, f"Candidate: {candidate_name}")
    c.drawString(40, height - 100, f"Job Title: {job_title}")
    c.drawString(40, height - 115, f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    c.drawString(40, height - 130, f"Method: {method_used}")
    c.drawString(40, height - 145, f"Overall Match Score: {match_score:.2f}%")

    # Pie chart image (smaller)
    if pie_fig is not None:
        imgbuf = io.BytesIO()
        pie_fig.savefig(imgbuf, bbox_inches="tight")
        imgbuf.seek(0)
        img = ImageReader(imgbuf)
        c.drawImage(img, width - 230, height - 260, width=200, height=200)

    # Gauge (matplotlib)
    gbuf = io.BytesIO()
    mg = matplotlib_gauge(match_score)
    mg.savefig(gbuf, bbox_inches="tight")
    plt.close(mg)
    gbuf.seek(0)
    img_g = ImageReader(gbuf)
    c.drawImage(img_g, width - 460, height - 260, width=200, height=160)

    # Matching top lines
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 170, "Top Matching Skills:")
    c.setFont("Helvetica", 10)
    y = height - 190
    for s in matching[:60]:
        c.drawString(45, y, f"- {s}")
        y -= 12
        if y < 120:
            c.showPage()
            y = height - 40

    # Suggestions
    if y < 160:
        c.showPage()
        y = height - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Top Suggestions:")
    y -= 18
    c.setFont("Helvetica", 10)
    for s in suggestions[:80]:
        c.drawString(45, y, f"- {s}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 40

    # Top sentences page
    c.showPage()
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, height - 60, "Top JD Sentences & Matched Resume Lines")
    y = height - 90
    c.setFont("Helvetica", 10)
    for ts in top_sentences:
        jd = ts.get("job_sentence","")[:180]
        mr = ts.get("best_resume_sentence","")[:180]
        sc = ts.get("score",0.0)
        c.drawString(40, y, f"JD: {jd}")
        y -= 12
        c.drawString(48, y, f"Matched: {mr} (score {sc:.2f})")
        y -= 18
        if y < 80:
            c.showPage()
            y = height - 40

    c.save()
    buffer.seek(0)
    return buffer.read()

# -----------------------
# Small curated resources for enhancements
# -----------------------
SKILL_RESOURCES = {
    "python": [{"title":"Python Docs","url":"https://docs.python.org/3/"}],
    "aws": [{"title":"AWS Training","url":"https://aws.amazon.com/training/"}],
    "docker":[{"title":"Docker Get Started","url":"https://www.docker.com/get-started"}],
    "react":[{"title":"React Tutorial","url":"https://reactjs.org/tutorial/tutorial.html"}],
    "nlp":[{"title":"NLP Specialization","url":"https://www.coursera.org/specializations/natural-language-processing"}]
}

def generate_resume_bullets(skill, context=None):
    return [
        f"Implemented {skill} in a project to {context or 'deliver business value'} and measured improvements.",
        f"Designed end-to-end {skill} pipeline including testing and deployment; documented outcomes and metrics."
    ]

# -----------------------
# UI pieces: signup/login/profile
# -----------------------
def signup_ui():
    st.subheader("Create account")
    col1, col2 = st.columns(2)
    username = col1.text_input("Username", key="su_username")
    email = col2.text_input("Email", key="su_email")
    full_name = st.text_input("Full name", key="su_fullname")
    password = st.text_input("Password", type="password", key="su_password")
    password2 = st.text_input("Confirm password", type="password", key="su_password2")
    avatar = st.file_uploader("Upload avatar (optional)", type=["png","jpg","jpeg"], key="su_avatar")
    bio = st.text_area("Short bio (optional)", max_chars=300, key="su_bio")
    if st.button("Create account"):
        if not username or not password:
            st.error("Username and password required.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return
        avatar_path = None
        if avatar:
            ext = os.path.splitext(avatar.name)[1]
            avatar_path = os.path.join(AVATARS_DIR, f"{username}_{uuid.uuid4().hex[:6]}{ext}")
            with open(avatar_path, "wb") as f:
                f.write(avatar.read())
        ok, msg = create_user(username.strip(), email.strip(), password, full_name.strip(), avatar_path, bio.strip())
        if ok:
            st.success("Account created. Please login from the sidebar.")
        else:
            st.error(msg)

def login_ui():
    st.subheader("Login")
    username = st.text_input("Username", key="li_username")
    password = st.text_input("Password", type="password", key="li_password")
    if st.button("Login"):
        uid = authenticate(username.strip(), password)
        if uid:
            st.session_state["user_id"] = uid
            st.session_state["username"] = username.strip()
            st.success("Logged in.")
        else:
            st.error("Invalid credentials.")

def profile_ui():
    st.header("Profile")
    uid = st.session_state["user_id"]
    user = get_user(uid)
    if not user:
        st.error("User not found.")
        return
    col1, col2 = st.columns([1,3])
    with col1:
        if user.get("avatar_path") and os.path.exists(user["avatar_path"]):
            st.image(user["avatar_path"], width=150)
        else:
            st.image("https://via.placeholder.com/150?text=Avatar", width=150)
        new_avatar = st.file_uploader("Change avatar", type=["png","jpg","jpeg"], key="pf_avatar")
        if new_avatar:
            ext = os.path.splitext(new_avatar.name)[1]
            av_path = os.path.join(AVATARS_DIR, f"{user['username']}_{uuid.uuid4().hex[:6]}{ext}")
            with open(av_path, "wb") as f:
                f.write(new_avatar.read())
            update_profile(uid, avatar_path=av_path)
            st.success("Avatar updated. Refresh to see change.")
    with col2:
        st.write(f"*Username:* {user['username']}")
        st.write(f"*Full name:* {user.get('full_name') or '—'}")
        st.write(f"*Email:* {user.get('email') or '—'}")
        st.write(f"*Bio:* {user.get('bio') or '—'}")
        full = st.text_input("Full name", value=user.get("full_name") or "")
        bio = st.text_area("Bio", value=user.get("bio") or "")
        if st.button("Update profile"):
            update_profile(uid, full_name=full.strip(), bio=bio.strip())
            st.success("Profile updated.")

# -----------------------
# Main UI and flow
# -----------------------
def main():
    init_database()
    st.set_page_config(page_title="Resume Analyzer — Polished", layout="wide")
    # Hero
    st.markdown("""
        <div style="background:linear-gradient(90deg,#0ea5e9,#3b82f6);padding:20px;border-radius:8px;color:white;margin-bottom:12px">
            <h1 style="margin:0">Resume Analyzer — Polished</h1>
            <p style="margin:4px 0 0 0;color:rgba(255,255,255,0.95)">Deep semantic matching, actionable resume suggestions, and polished PDF reports.</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar: auth + nav
    with st.sidebar:
        st.header("Account")
        if "user_id" in st.session_state:
            user = get_user(st.session_state["user_id"])
            st.write(f"Signed in as *{user['username']}*")
            if user.get("avatar_path") and os.path.exists(user["avatar_path"]):
                st.image(user["avatar_path"], width=100)
            if st.button("Logout"):
                for k in ["user_id", "username"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.success("Logged out.")
        else:
            choice = st.radio("Auth", ("Login", "Sign up"))
            if choice == "Login":
                login_ui()
            else:
                signup_ui()
        st.markdown("---")
        st.header("Navigate")
        page = st.radio("Go to", ["Home","Dashboard","My Analyses","Profile","About"], index=0)

    # Home
    if page == "Home":
        st.subheader("Analyze resume vs job description")
        col_left, col_right = st.columns([1,2])
        with col_left:
            resume_file = st.file_uploader("Upload Resume (pdf or docx)", type=["pdf","docx"])
            candidate_name = st.text_input("Candidate name", value=st.session_state.get("username",""))
            job_title = st.text_input("Target Job Title", "")
            save_profile = st.checkbox("Save to my account (requires login)", value=True)
            analysis_mode = st.selectbox("Analysis Mode", ["Advanced (recommended)","Basic (original)"])
            st.caption("Advanced mode uses semantic matching (Sentence-BERT) when available.")
        with col_right:
            job_desc = st.text_area("Paste Job Description", height=380)

        if resume_file and job_desc.strip():
            st.success("Analyzing — this may take a moment on first run (models load)...")
            resume_text = parse_resume(resume_file)
            resume_sections = extract_sections(resume_text)
            if analysis_mode == "Basic (original)":
                results = analyze_resume(resume_text, job_desc)  # original unchanged
                # filter out stopwords from results (do not change original function, but clean display)
                matching = [t for t in results["skills_match"] if t.lower() not in STOPWORDS and len(t)>2]
                missing = [t for t in results["missing_skills"] if t.lower() not in STOPWORDS and len(t)>2]
                suggestions = [s for s in results["suggestions"]]
                top_sentences = []
                match_score = (len(matching) / max(1, len(matching) + len(missing))) * 100.0
                method_used = "basic-original"
                breakdown = []
                enhancements = {}
            else:
                adv = analyze_resume_advanced(resume_text, job_desc, top_k_sentences=8)
                matching = adv["matching_skills"]
                missing = adv["missing_skills"]
                suggestions = adv["suggestions"]
                top_sentences = adv["top_sentences"]
                match_score = adv["match_score"]
                method_used = adv["method_used"]
                breakdown = adv.get("breakdown", [])
                enhancements = adv.get("enhancements", {})

            # Summary and gauge
            st.subheader("Summary")
            gcol, info_col = st.columns([1,1])
            with gcol:
                try:
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(plotly_gauge(match_score), use_container_width=True)
                    else:
                        st.pyplot(matplotlib_gauge(match_score))
                except Exception:
                    st.metric("Match Score", f"{match_score:.1f}%")
            with info_col:
                st.metric("Matching skills", len(matching))
                st.metric("Missing tokens", len(missing))
                st.write("Method used:", method_used)

            # Matching breakdown table
            st.subheader("Matching breakdown")
            if breakdown:
                df_break = pd.DataFrame(breakdown).sort_values("score", ascending=False)
                st.dataframe(df_break, height=220)
            else:
                st.write("Using exact token overlap in Basic mode.")

            # Matching chips (styled)
            st.subheader("Matching (top tokens)")
            if matching:
                chips = " ".join([f"<span style='display:inline-block;background:#e6f4ff;padding:6px 10px;border-radius:12px;margin:3px;font-size:12px'>{s}</span>" for s in matching[:120]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("No matching tokens detected.")

            # Missing / Enhancement Plan
            st.subheader("Missing skills & Enhancement plan")
            if not missing:
                st.write("No missing skills detected — great match!")
            else:
                # prioritized suggestions + bullets + resources
                for s in missing[:40]:
                    priority = enhancements.get(s, {}).get("priority", 0.0)
                    st.markdown(f"{s}** — priority: {priority:.3f}")
                    # bullets
                    bullets = enhancements.get(s, {}).get("bullets", generate_resume_bullets(s, context=job_title or "project"))
                    for b in bullets:
                        st.write("• " + b)
                    # resources
                    resources = SKILL_RESOURCES.get(s.lower(), [])
                    if resources:
                        st.write("Resources:")
                        for r in resources:
                            st.markdown(f"- [{r['title']}]({r['url']})")
                    st.markdown("---")

            # Top suggestions (short)
            st.subheader("Top suggestions")
            for i,s in enumerate(suggestions[:20], start=1):
                st.write(f"{i}. {s}")

            # Top JD sentences matched
            if top_sentences:
                st.subheader("Top job-description sentences covered")
                for ts in top_sentences:
                    st.write(f"- JD: {ts['job_sentence'][:220]}")
                    st.write(f"  Matched: {ts['best_resume_sentence'][:220]}  (score {ts['score']:.2f})")

            # Visuals: small pie
            st.subheader("Visual: Matching vs Missing")
            pie_fig = plot_skills_chart(matching, missing)
            st.pyplot(pie_fig)

            # Save file
            saved_path = os.path.join(UPLOADS_DIR, f"{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:6]}{resume_file.name}")
            with open(saved_path, "wb") as f:
                f.write(resume_file.getbuffer())

            # Save analysis if requested & logged-in
            if save_profile and "user_id" in st.session_state:
                uid = st.session_state["user_id"]
                cname = candidate_name or st.session_state.get("username","Candidate")
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""
                    INSERT INTO analyses (user_id, candidate_name, job_title, match_score, matching_skills, missing_skills, suggestions, top_sentences, created_at, resume_path, job_desc, method_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uid, cname, job_title or "Target Role", match_score,
                    json.dumps(matching, ensure_ascii=False), json.dumps(missing, ensure_ascii=False),
                    json.dumps(suggestions, ensure_ascii=False), json.dumps(top_sentences, ensure_ascii=False),
                    datetime.datetime.utcnow().isoformat(), saved_path, job_desc, method_used
                ))
                conn.commit()
                conn.close()
                st.success("Saved analysis to your account.")

            # PDF generation + download
            st.markdown("---")
            st.subheader("Export / Download")
            if st.button("Generate & Download PDF report"):
                pdf_bytes = generate_pdf_report(candidate_name or "Candidate", job_title or "Target Role",
                                                match_score, matching, missing, suggestions, top_sentences, method_used, pie_fig=pie_fig)
                st.download_button(label="Download report (PDF)", data=pdf_bytes, file_name="resume_analysis_report.pdf", mime="application/pdf")

        else:
            st.info("Please upload a resume and paste a job description to analyze (use Advanced mode for best results).")

    # Dashboard
    elif page == "Dashboard":
        if "user_id" not in st.session_state:
            st.warning("Login required.")
            return
        st.header("Dashboard")
        uid = st.session_state["user_id"]
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id,candidate_name,job_title,match_score,created_at FROM analyses WHERE user_id = ? ORDER BY created_at DESC", conn, params=(uid,))
        conn.close()
        if df.empty:
            st.info("No saved analyses yet.")
        else:
            df["created_at"] = pd.to_datetime(df["created_at"])
            st.dataframe(df, height=360)
            st.markdown("Load an analysis by ID to view details")
            min_id, max_id = int(df["id"].min()), int(df["id"].max())
            aid = st.number_input("Analysis ID", min_value=min_id, max_value=max_id, value=min_id, step=1)
            if st.button("Load"):
                rec = get_analysis_by_id(aid)
                if rec:
                    st.subheader(f"{rec['candidate_name']} — {rec['job_title']}")
                    st.write(f"Match: {rec['match_score']:.2f}%")
                    matching = json.loads(rec["matching_skills"] or "[]")
                    missing = json.loads(rec["missing_skills"] or "[]")
                    suggestions = json.loads(rec["suggestions"] or "[]")
                    top_sentences = json.loads(rec["top_sentences"] or "[]")
                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(plotly_gauge(rec['match_score']))
                    else:
                        st.pyplot(matplotlib_gauge(rec['match_score']))
                    st.write("Matching:", matching)
                    st.write("Missing:", missing)
                    st.write("Suggestions:", suggestions)
                    st.write("Top sentences:", top_sentences)
                    if rec.get("resume_path") and os.path.exists(rec["resume_path"]):
                        with open(rec["resume_path"], "rb") as f:
                            st.download_button("Download Resume", data=f, file_name=os.path.basename(rec["resume_path"]))

    # My Analyses
    elif page == "My Analyses":
        if "user_id" not in st.session_state:
            st.warning("Login required.")
            return
        uid = st.session_state["user_id"]
        st.header("My Analyses")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id,candidate_name,job_title,match_score,created_at FROM analyses WHERE user_id = ? ORDER BY created_at DESC", conn, params=(uid,))
        conn.close()
        if df.empty:
            st.info("No analyses saved.")
        else:
            for _, row in df.iterrows():
                st.write(f"{row['candidate_name']}** — {row['job_title']} — {row['match_score']:.1f}% — {row['created_at']}")
                if st.button(f"Load {row['id']}", key=f"load_{row['id']}"):
                    rec = get_analysis_by_id(int(row['id']))
                    if rec:
                        st.write(rec)

    # Profile
    elif page == "Profile":
        if "user_id" not in st.session_state:
            st.warning("Login required.")
            return
        profile_ui()

    # About
    else:
        st.header("About")
        st.write("This polished Resume Analyzer filters stopwords, uses semantic matching, and generates working PDF reports.")
        st.write("If you'd like, I can now add: LinkedIn/Adzuna job API, embedding caching in DB/Redis, background workers, or a React/Tailwind front-end.")

# -----------------------
# Small DB helper for loading a single analysis by id
# -----------------------
def get_analysis_by_id(aid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analyses WHERE id = ?", (aid,))
    row = c.fetchone()
    desc = c.description
    conn.close()
    if not row:
        return None
    keys = [d[0] for d in desc]
    return dict(zip(keys, row))

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    main()



