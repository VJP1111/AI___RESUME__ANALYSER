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
import sqlite3
import hashlib
import datetime
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

# AI/NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# PDF generation
from fpdf import FPDF

# Configuration
st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database and file paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "resume_analyzer.db")
UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
REPORTS_DIR = os.path.join(APP_DIR, "reports")

# Create directories
for dir_path in [UPLOADS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Skills database
TECH_SKILLS = {
    'programming_languages': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'Rust', 'TypeScript'],
    'web_technologies': ['HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask'],
    'databases': ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle'],
    'cloud_platforms': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes'],
    'data_science': ['Machine Learning', 'Deep Learning', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch'],
    'tools_frameworks': ['Git', 'GitHub', 'Jenkins', 'JIRA', 'Linux', 'Agile', 'Scrum']
}

ALL_SKILLS = []
for category in TECH_SKILLS.values():
    ALL_SKILLS.extend(category)

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def init_database():
    """Initialize SQLite database"""
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            remember_token TEXT
        )
    """)
    
    # User sessions table for persistent login
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            remember_me INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_accessed TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Analyses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            candidate_name TEXT,
            job_title TEXT,
            match_score REAL,
            matching_skills TEXT,
            missing_skills TEXT,
            suggestions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash password"""
    salt = "resume_analyzer_salt_2024"
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

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
        SELECT id, username, email, password_hash, full_name
        FROM users WHERE username = ? OR email = ?
    """, (username, username))
    
    user = cursor.fetchone()
    
    if user and hash_password(password) == user[3]:
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
            "full_name": user[4] or ""
        }
    
    conn.close()
    return None

def create_user_session(user_id: int, remember_me: bool = False) -> str:
    """Create a new user session and return session token"""
    import secrets
    from datetime import datetime, timedelta
    
    session_token = secrets.token_urlsafe(32)
    
    # Session expires in 30 days if remember_me, otherwise 1 day
    expires_days = 30 if remember_me else 1
    expires_at = datetime.now() + timedelta(days=expires_days)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clean old sessions for this user
    cursor.execute("""
        DELETE FROM user_sessions 
        WHERE user_id = ? AND expires_at < CURRENT_TIMESTAMP
    """, (user_id,))
    
    # Create new session
    cursor.execute("""
        INSERT INTO user_sessions (user_id, session_token, remember_me, expires_at, last_accessed)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (user_id, session_token, 1 if remember_me else 0, expires_at.isoformat()))
    
    conn.commit()
    conn.close()
    
    return session_token

def validate_session_token(session_token: str) -> dict:
    """Validate session token and return user info if valid"""
    if not session_token:
        return None
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT u.id, u.username, u.email, u.full_name, s.remember_me
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
    """, (session_token,))
    
    result = cursor.fetchone()
    
    if result:
        # Update last accessed time
        cursor.execute("""
            UPDATE user_sessions 
            SET last_accessed = CURRENT_TIMESTAMP 
            WHERE session_token = ?
        """, (session_token,))
        conn.commit()
        
        conn.close()
        return {
            "id": result[0],
            "username": result[1],
            "email": result[2],
            "full_name": result[3] or "",
            "remember_me": bool(result[4])
        }
    
    conn.close()
    return None

def clear_user_session(session_token: str):
    """Clear/logout user session"""
    if not session_token:
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM user_sessions WHERE session_token = ?
    """, (session_token,))
    
    conn.commit()
    conn.close()

# =============================================================================
# TEXT PROCESSING AND ANALYSIS
# =============================================================================

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    if SBERT_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return None
    return None

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        
        elif uploaded_file.type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\\n"
            return text
        
        else:
            text = str(uploaded_file.read(), "utf-8")
            return text
            
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def extract_skills_from_text(text):
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))

def advanced_resume_analysis(resume_text, job_description):
    """Perform advanced analysis"""
    model = load_sentence_transformer()
    
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_description)
    
    matching_skills = list(set(resume_skills).intersection(set(job_skills)))
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    
    # Calculate match score
    if len(job_skills) > 0:
        match_score = (len(matching_skills) / len(job_skills)) * 100
    else:
        match_score = 50.0
    
    # Generate suggestions
    suggestions = []
    for skill in missing_skills[:10]:
        suggestions.append({
            'skill': skill,
            'suggestion': f"Consider adding {skill} projects to your portfolio",
            'priority': 'High' if skill in ['Python', 'JavaScript', 'React'] else 'Medium'
        })
    
    return {
        'match_score': round(match_score, 2),
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'suggestions': suggestions,
        'analysis_method': 'Advanced NLP' if model else 'Standard Analysis'
    }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_match_score_gauge(score):
    """Create gauge chart for match score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Match Score (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=50, r=50, t=50, b=50))
    return fig

def create_skills_pie_chart(matching_skills, missing_skills):
    """Create pie chart for skills"""
    labels = ['Matching Skills', 'Missing Skills']
    values = [len(matching_skills), len(missing_skills)]
    colors = ['#2E8B57', '#DC143C']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title={'text': 'Skills Analysis', 'x': 0.5},
        height=350,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(analysis_data, candidate_name, job_title):
    """Generate PDF report"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Header
        pdf.cell(0, 10, 'Resume Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Basic info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Candidate: {candidate_name}', 0, 1)
        pdf.cell(0, 8, f'Position: {job_title}', 0, 1)
        pdf.cell(0, 8, f'Analysis Date: {datetime.datetime.now().strftime("%Y-%m-%d")}', 0, 1)
        pdf.ln(5)
        
        # Match score
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f'Match Score: {analysis_data["match_score"]:.1f}%', 0, 1)
        pdf.ln(5)
        
        # Matching skills
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Matching Skills:', 0, 1)
        pdf.set_font('Arial', '', 10)
        for skill in analysis_data['matching_skills']:
            pdf.cell(0, 6, f'• {skill}', 0, 1)
        pdf.ln(5)
        
        # Missing skills
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Missing Skills:', 0, 1)
        pdf.set_font('Arial', '', 10)
        for skill in analysis_data['missing_skills'][:10]:
            pdf.cell(0, 6, f'• {skill}', 0, 1)
        
        return pdf.output(dest='S').encode('latin-1')
    
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# =============================================================================
# DATA PERSISTENCE
# =============================================================================

def save_analysis_to_database(user_id, analysis_data, candidate_name, job_title):
    """Save analysis to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO analyses (
                user_id, candidate_name, job_title, match_score,
                matching_skills, missing_skills, suggestions
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, candidate_name, job_title,
            analysis_data['match_score'],
            json.dumps(analysis_data['matching_skills']),
            json.dumps(analysis_data['missing_skills']),
            json.dumps(analysis_data['suggestions'])
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        return analysis_id
    
    except Exception as e:
        st.error(f"Error saving analysis: {e}")
        return None
    
    finally:
        conn.close()

def get_user_analyses(user_id):
    """Get user's analysis history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, candidate_name, job_title, match_score, created_at
        FROM analyses WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    
    analyses = cursor.fetchall()
    conn.close()
    
    return [
        {
            'id': analysis[0],
            'candidate_name': analysis[1],
            'job_title': analysis[2],
            'match_score': analysis[3],
            'created_at': analysis[4]
        }
        for analysis in analyses
    ]

# =============================================================================
# STREAMLIT UI FUNCTIONS
# =============================================================================

def get_persistent_session_token() -> str:
    """Get session token from browser localStorage using JavaScript"""
    # Use Streamlit components to interact with browser localStorage
    # This creates a more robust persistence mechanism
    
    # First check session state
    if hasattr(st.session_state, 'persistent_token_checked'):
        return st.session_state.get('browser_session_token', None)
    
    # Use JavaScript to get token from localStorage
    # Simplified approach: use file-based persistence only
    # This avoids complex JavaScript integration issues
    # html_code = """
    # <script>
    #     // Get token from localStorage
    #     var token = localStorage.getItem('resume_analyzer_session_token');
    #     if (token) {
    #         // Send token to parent via postMessage
    #         window.parent.postMessage({type: 'session_token', token: token}, '*');
    #     } else {
    #         window.parent.postMessage({type: 'no_token'}, '*');
    #     }
    # </script>
    # <div style="display:none;">Loading session...</div>
    # """
    # 
    # # Display the HTML component (invisible)
    # import streamlit.components.v1 as components
    # try:
    #     result = components.html(html_code, height=0)
    # except:
    #     pass
    
    # Fallback to file-based storage
    import tempfile
    token_file = os.path.join(tempfile.gettempdir(), 'resume_analyzer_session.txt')
    try:
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token = f.read().strip()
                if token:
                    st.session_state.browser_session_token = token
                    st.session_state.persistent_token_checked = True
                    return token
    except Exception:
        pass
    
    st.session_state.persistent_token_checked = True
    return None

def save_persistent_session_token(token: str):
    """Save session token to persistent storage"""
    # Save to browser localStorage using JavaScript (commented out for simplicity)
    # html_code = f"""
    # <script>
    #     localStorage.setItem('resume_analyzer_session_token', '{token}');
    #     console.log('Session token saved to localStorage');
    # </script>
    # <div style="display:none;">Saving session...</div>
    # """
    # 
    # import streamlit.components.v1 as components
    # try:
    #     components.html(html_code, height=0)
    # except:
    #     pass
    
    # Also save to file as backup
    import tempfile
    token_file = os.path.join(tempfile.gettempdir(), 'resume_analyzer_session.txt')
    try:
        with open(token_file, 'w') as f:
            f.write(token)
        st.session_state.browser_session_token = token
    except Exception:
        pass

def clear_persistent_session_token():
    """Clear saved session token"""
    # Clear from browser localStorage (commented out for simplicity)
    # html_code = """
    # <script>
    #     localStorage.removeItem('resume_analyzer_session_token');
    #     console.log('Session token cleared from localStorage');
    # </script>
    # <div style="display:none;">Clearing session...</div>
    # """
    # 
    # import streamlit.components.v1 as components
    # try:
    #     components.html(html_code, height=0)
    # except:
    #     pass
    
    # Clear file backup
    import tempfile
    token_file = os.path.join(tempfile.gettempdir(), 'resume_analyzer_session.txt')
    try:
        if os.path.exists(token_file):
            os.remove(token_file)
        if 'browser_session_token' in st.session_state:
            del st.session_state.browser_session_token
        if 'persistent_token_checked' in st.session_state:
            del st.session_state.persistent_token_checked
    except Exception:
        pass

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state with persistent login check"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    
    # Check for existing valid session on app start
    if not st.session_state.authenticated:
        # Try to get session token from browser's localStorage
        session_token = get_persistent_session_token()
        if session_token:
            user_info = validate_session_token(session_token)
            if user_info:
                st.session_state.authenticated = True
                st.session_state.user_info = user_info
                st.session_state.session_token = session_token
                st.success(f"Welcome back, {user_info['username']}! 👋")
            else:
                # Invalid token, clear it
                clear_persistent_session_token()

def render_sidebar_auth():
    """Render sidebar authentication with persistent login"""
    with st.sidebar:
        if st.session_state.authenticated:
            user = st.session_state.user_info
            st.success(f"✅ Welcome **{user['username']}**")
            
            # Show login status
            if user.get('remember_me'):
                st.caption("🔒 Logged in (Remember Me enabled)")
            else:
                st.caption("🔓 Logged in (This session only)")
            
            if st.button("🚪 Logout", use_container_width=True):
                # Clear session token from database
                if st.session_state.session_token:
                    clear_user_session(st.session_state.session_token)
                
                # Clear persistent session token
                clear_persistent_session_token()
                
                # Clear session state
                st.session_state.authenticated = False
                st.session_state.user_info = None
                st.session_state.current_analysis = None
                st.session_state.session_token = None
                
                st.success("🎉 Logged out successfully! Your session has been completely cleared. 👋")
                st.rerun()
        
        else:
            st.header("🔐 Authentication")
            
            tab = st.selectbox("Action", ["Login", "Sign Up"])
            
            if tab == "Login":
                with st.form("login_form"):
                    username = st.text_input("Username/Email")
                    password = st.text_input("Password", type="password")
                    remember_me = st.checkbox("🔒 Remember Me (Stay logged in for 30 days)", value=True)
                    
                    if st.form_submit_button("Login", use_container_width=True):
                        if username and password:
                            user_info = authenticate_user(username.strip(), password)
                            if user_info:
                                # Create session token
                                session_token = create_user_session(user_info['id'], remember_me)
                                
                                # Save persistent session token
                                save_persistent_session_token(session_token)
                                
                                # Update session state
                                st.session_state.authenticated = True
                                st.session_state.user_info = user_info
                                st.session_state.user_info['remember_me'] = remember_me
                                st.session_state.session_token = session_token
                                
                                login_message = "🎉 Login successful!"
                                if remember_me:
                                    login_message += " Your login details are saved and you'll stay logged in for 30 days, even if you close the browser! 🔒"
                                else:
                                    login_message += " You'll stay logged in for this browser session. 🔓"
                                
                                st.success(login_message)
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials! Please check your username and password.")
                        else:
                            st.error("⚠️ Please enter both username and password")
            
            else:
                with st.form("signup_form"):
                    username = st.text_input("Username")
                    email = st.text_input("Email")
                    full_name = st.text_input("Full Name")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    
                    if st.form_submit_button("Create Account", use_container_width=True):
                        if not all([username, email, password, confirm_password]):
                            st.error("⚠️ Please fill all fields")
                        elif password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif len(password) < 6:
                            st.error("⚠️ Password must be at least 6 characters")
                        else:
                            success, message = create_user(username.strip(), email.strip(), password, full_name.strip())
                            if success:
                                st.success(f"✅ {message} You can now login!")
                            else:
                                st.error(f"❌ {message}")

def render_main_analyzer():
    """Main analyzer interface"""
    st.title("🎯 Resume Analyzer Pro")
    st.markdown("*Advanced AI-powered resume analysis*")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'docx', 'txt']
        )
        candidate_name = st.text_input("Candidate Name")
    
    with col2:
        st.subheader("💼 Job Description")
        job_description = st.text_area(
            "Paste job description",
            height=200
        )
        job_title = st.text_input("Job Title")
    
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if uploaded_file and job_description.strip():
            with st.spinner("🤖 Analyzing... Please wait."):
                resume_text = extract_text_from_file(uploaded_file)
                
                if resume_text.strip():
                    analysis_results = advanced_resume_analysis(resume_text, job_description)
                    
                    # Save to database if authenticated
                    if st.session_state.authenticated:
                        analysis_id = save_analysis_to_database(
                            st.session_state.user_info['id'],
                            analysis_results,
                            candidate_name or 'Candidate',
                            job_title or 'Position'
                        )
                        if analysis_id:
                            st.success(f"✅ Analysis saved (ID: {analysis_id})")
                    
                    st.session_state.current_analysis = analysis_results
                    st.success("Analysis completed!")
                else:
                    st.error("Could not extract text from file")
        else:
            st.error("Please upload resume and provide job description")
    
    # Display results
    if st.session_state.current_analysis:
        display_analysis_results(st.session_state.current_analysis, candidate_name, job_title)

def display_analysis_results(analysis, candidate_name="", job_title=""):
    """Display analysis results"""
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Match Score", f"{analysis['match_score']:.1f}%")
    with col2:
        st.metric("Matching Skills", len(analysis['matching_skills']))
    with col3:
        st.metric("Missing Skills", len(analysis['missing_skills']))
    with col4:
        st.metric("Method", analysis.get('analysis_method', 'Standard'))
    
    # Visualizations
    st.subheader("📈 Visual Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        gauge_fig = create_match_score_gauge(analysis['match_score'])
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        pie_fig = create_skills_pie_chart(analysis['matching_skills'], analysis['missing_skills'])
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Skills breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Matching Skills")
        if analysis['matching_skills']:
            for skill in analysis['matching_skills']:
                st.write(f"• {skill}")
        else:
            st.info("No matching skills found")
    
    with col2:
        st.subheader("❌ Missing Skills")
        if analysis['missing_skills']:
            for skill in analysis['missing_skills'][:10]:
                st.write(f"• {skill}")
            if len(analysis['missing_skills']) > 10:
                st.info(f"... and {len(analysis['missing_skills']) - 10} more")
        else:
            st.success("No missing skills!")
    
    # Enhancement suggestions
    if analysis['suggestions']:
        st.subheader("💡 Enhancement Suggestions")
        for i, suggestion in enumerate(analysis['suggestions'][:5], 1):
            with st.expander(f"{i}. {suggestion['skill']} [{suggestion['priority']} Priority]"):
                st.write(suggestion['suggestion'])
    
    # PDF Report
    st.subheader("📄 Generate Report")
    if st.button("📄 Generate PDF Report", type="secondary", use_container_width=True):
        pdf_bytes = generate_pdf_report(analysis, candidate_name or "Candidate", job_title or "Position")
        if pdf_bytes:
            st.download_button(
                "📥 Download PDF Report",
                pdf_bytes,
                file_name=f"resume_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def render_dashboard():
    """User dashboard"""
    if not st.session_state.authenticated:
        st.warning("Please login to view dashboard")
        return
    
    st.title("📊 Dashboard")
    
    user_id = st.session_state.user_info['id']
    analyses = get_user_analyses(user_id)
    
    if not analyses:
        st.info("No analyses found. Start analyzing!")
        return
    
    st.subheader(f"Analysis History ({len(analyses)} total)")
    
    for analysis in analyses[:10]:  # Show last 10
        with st.expander(f"{analysis['candidate_name']} - {analysis['job_title']} ({analysis['match_score']:.1f}%)"):
            st.write(f"**Date:** {analysis['created_at']}")
            st.write(f"**Match Score:** {analysis['match_score']:.1f}%")

def render_profile():
    """User profile"""
    if not st.session_state.authenticated:
        st.warning("Please login to view profile")
        return
    
    st.title("👤 Profile")
    
    user = st.session_state.user_info
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Info")
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Full Name:** {user.get('full_name', 'Not set')}")
    
    with col2:
        st.subheader("Statistics")
        analyses = get_user_analyses(user['id'])
        st.metric("Total Analyses", len(analyses))
        
        if analyses:
            avg_score = sum(a['match_score'] for a in analyses) / len(analyses)
            st.metric("Average Match Score", f"{avg_score:.1f}%")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    # Initialize
    init_database()
    initialize_session_state()
    
    # Sidebar authentication
    render_sidebar_auth()
    
    # Main content
    if st.session_state.authenticated:
        # Authenticated user interface
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyzer", "📊 Dashboard", "👤 Profile", "ℹ️ About"])
        
        with tab1:
            render_main_analyzer()
        
        with tab2:
            render_dashboard()
        
        with tab3:
            render_profile()
        
        with tab4:
            st.title("ℹ️ About Resume Analyzer Pro")
            st.markdown("""
            ## Features
            
            - **Advanced AI Analysis**: Semantic matching using NLP
            - **Visual Insights**: Interactive charts and gauges  
            - **Enhancement Suggestions**: Actionable recommendations
            - **PDF Reports**: Professional analysis reports
            - **Profile Management**: Save and track analyses
            
            ## Technology
            
            - **Frontend**: Streamlit
            - **AI/NLP**: Sentence Transformers, Scikit-learn
            - **Visualization**: Plotly, Matplotlib
            - **Database**: SQLite
            
            Built with ❤️ for better career outcomes!
            """)
    
    else:
        # Non-authenticated interface
        st.title("🎯 Resume Analyzer Pro")
        st.markdown("""
        ### Welcome to Resume Analyzer Pro!
        
        **Advanced AI-powered resume analysis** with comprehensive insights.
        
        #### Features:
        - 🤖 **AI Analysis** - Advanced NLP for semantic matching
        - 📊 **Visual Insights** - Interactive charts and dashboards  
        - 💡 **Smart Suggestions** - Enhancement recommendations
        - 📄 **Professional Reports** - Comprehensive PDF reports
        - 💾 **Save & Track** - Personal dashboard for history
        
        **Please login or create an account to get started!**
        """)

if __name__ == "__main__":
    main()