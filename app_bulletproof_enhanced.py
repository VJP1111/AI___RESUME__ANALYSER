# # =============================================================================
# # 🚀 NEXGEN AI RESUME ANALYZER - BULLETPROOF ENHANCED VERSION
# # World's Most Advanced Career Platform - All Issues Fixed
# # =============================================================================

# import streamlit as st
# import os
# import sqlite3
# import hashlib
# import datetime
# import json
# import secrets
# import tempfile
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Any
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import io
# import base64

# # Document Processing with bulletproof imports
# import PyPDF2
# import docx

# # Advanced AI/NLP with fallback handling
# try:
#     from sentence_transformers import SentenceTransformer
#     from sklearn.metrics.pairwise import cosine_similarity
#     from textblob import TextBlob
#     AI_ENABLED = True
# except ImportError:
#     AI_ENABLED = False
#     st.info("⚠️ Advanced AI features require: pip install sentence-transformers scikit-learn textblob")

# # PDF Report Generation
# try:
#     from reportlab.lib.pagesizes import letter, A4
#     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
#     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
#     from reportlab.lib.units import inch
#     from reportlab.lib import colors
#     PDF_ENABLED = True
# except ImportError:
#     PDF_ENABLED = False
#     st.info("⚠️ PDF reports require: pip install reportlab")

# # OpenAI Integration for LLM
# try:
#     import openai
#     OPENAI_ENABLED = True
# except ImportError:
#     OPENAI_ENABLED = False
#     st.info("⚠️ Advanced LLM features require: pip install openai")

# # Job API Integration
# import requests
# from urllib.parse import quote

# # Enhanced UI Configuration
# st.set_page_config(
#     page_title="NexGen AI Resume Analyzer - Bulletproof",
#     page_icon="🚀", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Professional Styling
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
#     .nexgen-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem; border-radius: 15px; color: white; text-align: center;
#         margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#     }
    
#     .ai-metric {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
#         margin: 0.5rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.15);
#     }
    
#     .skill-tag {
#         background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
#         color: white; padding: 0.4rem 1rem; border-radius: 25px;
#         margin: 0.3rem; display: inline-block; font-size: 0.9rem;
#         box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Enhanced Configuration
# class Config:
#     DB_PATH = Path(__file__).parent / "nexgen_ai_analyzer.db"
#     UPLOADS_DIR = Path(__file__).parent / "uploads"
#     REPORTS_DIR = Path(__file__).parent / "reports"
    
#     # API Keys (set these in environment or here)
#     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
#     ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID', 'your-adzuna-app-id')
#     ADZUNA_API_KEY = os.getenv('ADZUNA_API_KEY', 'your-adzuna-api-key')
    
#     for dir_path in [UPLOADS_DIR, REPORTS_DIR]:
#         try:
#             dir_path.mkdir(exist_ok=True)
#         except:
#             pass

# # Comprehensive Skills Database
# NEXGEN_SKILLS = {
#     'AI/ML': ['Machine Learning', 'Deep Learning', 'Neural Networks', 'Computer Vision', 
#               'Natural Language Processing', 'TensorFlow', 'PyTorch', 'GPT', 'BERT'],
#     'Programming': ['Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go', 'Rust'],
#     'Cloud/DevOps': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins'],
#     'Web Dev': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Next.js'],
#     'Data Science': ['Pandas', 'NumPy', 'Matplotlib', 'SQL', 'Tableau', 'Apache Spark']
# }

# ALL_SKILLS = [skill for category in NEXGEN_SKILLS.values() for skill in category]

# # =============================================================================
# # DATABASE MANAGEMENT
# # =============================================================================

# def init_nexgen_database():
#     """Initialize bulletproof database"""
#     try:
#         conn = sqlite3.connect(Config.DB_PATH)
#         cursor = conn.cursor()
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 username TEXT UNIQUE NOT NULL,
#                 email TEXT UNIQUE NOT NULL,
#                 password_hash TEXT NOT NULL,
#                 salt TEXT NOT NULL,
#                 full_name TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 total_analyses INTEGER DEFAULT 0
#             )
#         """)
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS user_sessions (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 user_id INTEGER NOT NULL,
#                 session_token TEXT UNIQUE NOT NULL,
#                 remember_me INTEGER DEFAULT 0,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 expires_at TIMESTAMP,
#                 FOREIGN KEY (user_id) REFERENCES users (id)
#             )
#         """)
        
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS ai_analyses (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 user_id INTEGER NOT NULL,
#                 candidate_name TEXT,
#                 job_title TEXT,
#                 overall_match_score REAL,
#                 matching_skills TEXT,
#                 missing_skills TEXT,
#                 ai_insights TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (user_id) REFERENCES users (id)
#             )
#         """)
        
#         conn.commit()
#         conn.close()
#         return True
        
#     except Exception as e:
#         st.error(f"Database error: {e}")
#         return False

# init_nexgen_database()

# # =============================================================================
# # ENHANCED AI ENGINE
# # =============================================================================

# class NexGenAI:
#     def __init__(self):
#         self.model = None
#         self.load_ai_models()
    
#     def load_ai_models(self):
#         if AI_ENABLED:
#             try:
#                 self.model = SentenceTransformer('all-MiniLM-L6-v2')
#                 st.success("🤖 NexGen AI Engine Loaded Successfully!")
#             except:
#                 st.warning("⚠️ Using fallback AI mode")
    
#     def extract_skills_ai(self, text: str) -> Dict[str, Any]:
#         """Enhanced skill extraction"""
#         if not text:
#             return {"skills": [], "confidence": 0.0}
        
#         found_skills = []
#         text_lower = text.lower()
        
#         if self.model:
#             try:
#                 text_embedding = self.model.encode([text_lower])
#                 skill_embeddings = self.model.encode(ALL_SKILLS)
#                 similarities = cosine_similarity(text_embedding, skill_embeddings)[0]
                
#                 for skill, sim in zip(ALL_SKILLS, similarities):
#                     if sim > 0.7 or skill.lower() in text_lower:
#                         found_skills.append(skill)
#             except:
#                 found_skills = [skill for skill in ALL_SKILLS if skill.lower() in text_lower]
#         else:
#             found_skills = [skill for skill in ALL_SKILLS if skill.lower() in text_lower]
        
#         return {
#             "skills": list(set(found_skills)),
#             "confidence": 0.85 if self.model else 0.6
#         }
    
#     def analyze_resume_job_match(self, resume_text: str, job_desc: str) -> Dict[str, Any]:
#         """Comprehensive AI analysis"""
#         if not resume_text or not job_desc:
#             return self._empty_analysis()
        
#         try:
#             resume_skills = self.extract_skills_ai(resume_text)
#             job_skills = self.extract_skills_ai(job_desc)
            
#             resume_set = set(s.lower() for s in resume_skills["skills"])
#             job_set = set(s.lower() for s in job_skills["skills"])
            
#             matching = list(resume_set.intersection(job_set))
#             missing = list(job_set - resume_set)
            
#             if self.model:
#                 try:
#                     resume_emb = self.model.encode([resume_text])
#                     job_emb = self.model.encode([job_desc])
#                     semantic_score = cosine_similarity(resume_emb, job_emb)[0][0] * 100
#                 except:
#                     semantic_score = len(matching) / max(len(job_set), 1) * 100
#             else:
#                 semantic_score = len(matching) / max(len(job_set), 1) * 100
            
#             skill_score = len(matching) / max(len(job_set), 1) * 100
#             overall_score = (semantic_score * 0.4 + skill_score * 0.6)
            
#             insights = self.generate_ai_insights(resume_text)
            
#             return {
#                 "overall_match_score": round(overall_score, 2),
#                 "semantic_score": round(semantic_score, 2),
#                 "skill_score": round(skill_score, 2),
#                 "matching_skills": [s.title() for s in matching],
#                 "missing_skills": [s.title() for s in missing],
#                 "ai_insights": insights,
#                 "enhancement_suggestions": self._generate_suggestions(missing)
#             }
            
#         except Exception as e:
#             st.error(f"Analysis error: {e}")
#             return self._empty_analysis()
    
#     def generate_ai_insights(self, text: str) -> Dict[str, Any]:
#         """Generate AI insights with LLM integration"""
#         insights = {
#             "summary": "Professional with strong background",
#             "strengths": [],
#             "recommendations": [],
#             "sentiment": {},
#             "llm_analysis": ""
#         }
        
#         if text:
#             try:
#                 # Enhanced sentiment analysis
#                 if AI_ENABLED:
#                     blob = TextBlob(text)
#                     insights["sentiment"] = {
#                         "polarity": round(blob.sentiment.polarity, 3),
#                         "tone": "Positive" if blob.sentiment.polarity > 0 else "Professional"
#                     }
                
#                 # Rule-based strengths detection
#                 if any(word in text.lower() for word in ['lead', 'manage', 'direct']):
#                     insights["strengths"].append("Leadership experience")
                
#                 if any(word in text.lower() for word in ['python', 'machine learning', 'ai']):
#                     insights["strengths"].append("Strong technical skills")
                
#                 if any(word in text.lower() for word in ['project', 'delivered', 'implemented']):
#                     insights["strengths"].append("Project management")
                
#                 # LLM Analysis with OpenAI
#                 if OPENAI_ENABLED and Config.OPENAI_API_KEY != 'your-openai-api-key-here':
#                     try:
#                         openai.api_key = Config.OPENAI_API_KEY
                        
#                         response = openai.ChatCompletion.create(
#                             model="gpt-3.5-turbo",
#                             messages=[
#                                 {"role": "system", "content": "You are an expert HR analyst. Analyze the resume and provide professional insights."},
#                                 {"role": "user", "content": f"Analyze this resume and provide key insights:\n\n{text[:2000]}"}
#                             ],
#                             max_tokens=300,
#                             temperature=0.3
#                         )
                        
#                         insights["llm_analysis"] = response.choices[0].message.content
                        
#                     except Exception as e:
#                         insights["llm_analysis"] = f"LLM analysis unavailable: {str(e)[:100]}"
                
#                 insights["recommendations"] = [
#                     "Add quantifiable achievements",
#                     "Include relevant certifications", 
#                     "Highlight key projects",
#                     "Use action verbs to describe experience"
#                 ]
                
#             except Exception as e:
#                 st.warning(f"Insights generation error: {e}")
        
#         return insights
    
#     def _generate_suggestions(self, missing_skills: List[str]) -> List[Dict[str, Any]]:
#         """Generate enhancement suggestions"""
#         suggestions = []
#         for skill in missing_skills[:5]:
#             suggestions.append({
#                 "skill": skill.title(),
#                 "priority": "High" if skill.lower() in ['python', 'aws'] else "Medium",
#                 "action": f"Learn {skill.title()} through online courses"
#             })
#         return suggestions
    
#     def _empty_analysis(self) -> Dict[str, Any]:
#         return {
#             "overall_match_score": 0.0,
#             "semantic_score": 0.0, 
#             "skill_score": 0.0,
#             "matching_skills": [],
#             "missing_skills": [],
#             "ai_insights": {},
#             "enhancement_suggestions": []
#         }

# nexgen_ai = NexGenAI()

# # =============================================================================
# # JOB RECOMMENDATION ENGINE WITH REAL APIS
# # =============================================================================

# class JobRecommendationEngine:
#     def __init__(self):
#         self.adzuna_base_url = "https://api.adzuna.com/v1/api/jobs"
#         self.headers = {'User-Agent': 'NexGen AI Resume Analyzer'}
    
#     def search_jobs_adzuna(self, skills: List[str], country: str = "us", location: str = "", max_results: int = 10) -> List[Dict[str, Any]]:
#         """Search jobs using Adzuna API with global country support"""
#         try:
#             if Config.ADZUNA_APP_ID == 'your-adzuna-app-id' or Config.ADZUNA_API_KEY == 'your-adzuna-api-key':
#                 return self._get_sample_jobs(skills, country)
            
#             query = " OR ".join(skills[:5])  # Use top 5 skills
#             url = f"{self.adzuna_base_url}/{country}/search/1"
            
#             params = {
#                 'app_id': Config.ADZUNA_APP_ID,
#                 'app_key': Config.ADZUNA_API_KEY,
#                 'results_per_page': max_results,
#                 'what': query,
#                 'content-type': 'application/json'
#             }
            
#             if location:
#                 params['where'] = location
            
#             response = requests.get(url, params=params, headers=self.headers, timeout=15)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 jobs = []
                
#                 for job in data.get('results', []):
#                     jobs.append({
#                         'title': job.get('title', 'N/A'),
#                         'company': job.get('company', {}).get('display_name', 'N/A'),
#                         'location': job.get('location', {}).get('display_name', 'N/A'),
#                         'salary': self._format_salary(job.get('salary_min'), job.get('salary_max')),
#                         'description': job.get('description', '')[:300] + '...',
#                         'url': job.get('redirect_url', '#'),
#                         'created': job.get('created', 'N/A'),
#                         'match_score': self._calculate_job_match(job.get('description', ''), skills),
#                         'country': country.upper()
#                     })
                
#                 return sorted(jobs, key=lambda x: x['match_score'], reverse=True)
#             else:
#                 st.warning(f"API returned status {response.status_code}. Using sample data.")
            
#         except Exception as e:
#             st.warning(f"Job search error: {e}. Using sample data.")
        
#         return self._get_sample_jobs(skills, country)
    
#     def search_jobs_remotive(self, skills: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
#         """Search remote jobs using Remotive API"""
#         try:
#             url = "https://remotive.io/api/remote-jobs"
#             params = {'limit': max_results}
            
#             response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 jobs = []
                
#                 for job in data.get('jobs', [])[:max_results]:
#                     # Filter jobs that match skills
#                     job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
#                     skill_matches = sum(1 for skill in skills if skill.lower() in job_text)
                    
#                     if skill_matches > 0:
#                         jobs.append({
#                             'title': job.get('title', 'N/A'),
#                             'company': job.get('company_name', 'N/A'),
#                             'location': 'Remote',
#                             'salary': 'Competitive',
#                             'description': job.get('description', '')[:200] + '...',
#                             'url': job.get('url', '#'),
#                             'created': job.get('publication_date', 'N/A'),
#                             'match_score': (skill_matches / len(skills)) * 100
#                         })
                
#                 return sorted(jobs, key=lambda x: x['match_score'], reverse=True)
            
#         except Exception as e:
#             st.warning(f"Remote job search error: {e}")
        
#         return []
    
#     def _calculate_job_match(self, job_description: str, skills: List[str]) -> float:
#         """Calculate job match score based on skills"""
#         if not job_description or not skills:
#             return 0.0
        
#         job_text = job_description.lower()
#         matches = sum(1 for skill in skills if skill.lower() in job_text)
#         return (matches / len(skills)) * 100
    
#     def _format_salary(self, min_sal, max_sal):
#         """Format salary range"""
#         if min_sal and max_sal:
#             return f"${min_sal:,} - ${max_sal:,}"
#         elif min_sal:
#             return f"${min_sal:,}+"
#         else:
#             return "Competitive"
    
#     def _get_sample_jobs(self, skills: List[str], country: str = "us") -> List[Dict[str, Any]]:
#         """Return sample jobs when API is not available - customized by country"""
        
#         # Country-specific sample jobs
#         country_jobs = {
#             "us": [
#                 {
#                     'title': 'Senior Software Engineer',
#                     'company': 'Tech Innovations Inc.',
#                     'location': 'San Francisco, CA',
#                     'salary': '$120,000 - $160,000',
#                     'description': 'Join our team building cutting-edge applications using Python, React, and AWS. We offer competitive benefits and remote work options...',
#                     'url': 'https://example.com/job1',
#                     'created': '2024-01-15',
#                     'match_score': 85.0,
#                     'country': 'US'
#                 },
#                 {
#                     'title': 'Data Scientist',
#                     'company': 'AI Solutions Corp',
#                     'location': 'New York, NY',
#                     'salary': '$100,000 - $140,000',
#                     'description': 'Apply machine learning and data analysis to solve complex business problems. Experience with Python, TensorFlow, and SQL required...',
#                     'url': 'https://example.com/job2',
#                     'created': '2024-01-14',
#                     'match_score': 78.0,
#                     'country': 'US'
#                 }
#             ],
#             "gb": [
#                 {
#                     'title': 'Full Stack Developer',
#                     'company': 'London Tech Ltd',
#                     'location': 'London, UK',
#                     'salary': '£50,000 - £70,000',
#                     'description': 'Build innovative web applications using React, Node.js, and MongoDB. Join our dynamic team in the heart of London...',
#                     'url': 'https://example.com/job3',
#                     'created': '2024-01-13',
#                     'match_score': 82.0,
#                     'country': 'UK'
#                 },
#                 {
#                     'title': 'DevOps Engineer',
#                     'company': 'CloudFirst Solutions',
#                     'location': 'Manchester, UK',
#                     'salary': '£45,000 - £65,000',
#                     'description': 'Manage cloud infrastructure using AWS, Docker, and Kubernetes. Experience with CI/CD pipelines essential...',
#                     'url': 'https://example.com/job4',
#                     'created': '2024-01-12',
#                     'match_score': 75.0,
#                     'country': 'UK'
#                 }
#             ],
#             "ca": [
#                 {
#                     'title': 'Machine Learning Engineer',
#                     'company': 'AI Canada Inc.',
#                     'location': 'Toronto, ON',
#                     'salary': 'CAD $90,000 - $120,000',
#                     'description': 'Develop ML models using Python, TensorFlow, and PyTorch. Work on cutting-edge AI projects in healthcare and finance...',
#                     'url': 'https://example.com/job5',
#                     'created': '2024-01-11',
#                     'match_score': 88.0,
#                     'country': 'CA'
#                 }
#             ],
#             "au": [
#                 {
#                     'title': 'Software Developer',
#                     'company': 'Sydney Tech Hub',
#                     'location': 'Sydney, NSW',
#                     'salary': 'AUD $80,000 - $110,000',
#                     'description': 'Join our agile development team working with React, Python, and AWS. Great work-life balance and beautiful office location...',
#                     'url': 'https://example.com/job6',
#                     'created': '2024-01-10',
#                     'match_score': 79.0,
#                     'country': 'AU'
#                 }
#             ],
#             "de": [
#                 {
#                     'title': 'Backend Developer',
#                     'company': 'Berlin Innovations',
#                     'location': 'Berlin, Germany',
#                     'salary': '€55,000 - €75,000',
#                     'description': 'Develop scalable backend systems using Java, Spring Boot, and microservices architecture. Excellent benefits package...',
#                     'url': 'https://example.com/job7',
#                     'created': '2024-01-09',
#                     'match_score': 81.0,
#                     'country': 'DE'
#                 }
#             ]
#         }
        
#         sample_jobs = country_jobs.get(country.lower(), country_jobs["us"])
        
#         # Filter jobs based on user skills
#         filtered_jobs = []
#         for job in sample_jobs:
#             job_skills = job['description'].lower()
#             matches = sum(1 for skill in skills if skill.lower() in job_skills)
#             if matches > 0:
#                 job['match_score'] = min((matches / len(skills)) * 100, 95.0)
#                 filtered_jobs.append(job)
        
#         # Add more generic jobs if not enough matches
#         if len(filtered_jobs) < 3:
#             filtered_jobs.extend(sample_jobs[:3])
        
#         return filtered_jobs[:8]

# # Initialize Job Engine
# job_engine = JobRecommendationEngine()

# # =============================================================================
# # AUTHENTICATION
# # =============================================================================

# class NexGenAuth:
#     @staticmethod
#     def hash_password(password: str, salt: str) -> str:
#         return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
    
#     @staticmethod
#     def create_user(username: str, email: str, password: str, full_name: str = "") -> Tuple[bool, str]:
#         if len(password) < 6:
#             return False, "Password must be at least 6 characters"
        
#         try:
#             conn = sqlite3.connect(Config.DB_PATH)
#             cursor = conn.cursor()
            
#             salt = secrets.token_hex(32)
#             password_hash = NexGenAuth.hash_password(password, salt)
            
#             cursor.execute("""
#                 INSERT INTO users (username, email, password_hash, salt, full_name)
#                 VALUES (?, ?, ?, ?, ?)
#             """, (username, email, password_hash, salt, full_name))
            
#             conn.commit()
#             conn.close()
#             return True, "Account created successfully!"
            
#         except sqlite3.IntegrityError:
#             return False, "Username or email already exists"
#         except Exception as e:
#             return False, f"Error: {e}"
    
#     @staticmethod
#     def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
#         try:
#             conn = sqlite3.connect(Config.DB_PATH)
#             cursor = conn.cursor()
            
#             cursor.execute("""
#                 SELECT id, username, email, password_hash, salt, full_name
#                 FROM users WHERE username = ? OR email = ?
#             """, (username, username))
            
#             user = cursor.fetchone()
            
#             if user and NexGenAuth.hash_password(password, user[4]) == user[3]:
#                 conn.close()
#                 return {
#                     "id": user[0],
#                     "username": user[1],
#                     "email": user[2],
#                     "full_name": user[5] or ""
#                 }
            
#             conn.close()
#             return None
            
#         except Exception as e:
#             return None

# # =============================================================================
# # DOCUMENT PROCESSING
# # =============================================================================

# def extract_text_from_file(uploaded_file) -> str:
#     """Bulletproof text extraction"""
#     try:
#         if uploaded_file.type == "application/pdf":
#             reader = PyPDF2.PdfReader(uploaded_file)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text()
#             return text
        
#         elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
#             doc = docx.Document(uploaded_file)
#             return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
#         else:
#             return uploaded_file.read().decode('utf-8')
    
#     except Exception as e:
#         st.error(f"Error extracting text: {e}")
#         return ""

# # =============================================================================
# # PDF REPORT GENERATION
# # =============================================================================

# def generate_pdf_report(analysis: Dict[str, Any], candidate_name: str, job_title: str) -> bytes:
#     """Generate professional PDF report with enhanced content"""
#     if not PDF_ENABLED:
#         st.error("PDF generation requires reportlab package. Install with: pip install reportlab")
#         return b""
    
#     try:
#         buffer = io.BytesIO()
#         doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
#         styles = getSampleStyleSheet()
#         story = []
        
#         # Custom styles
#         title_style = ParagraphStyle(
#             'CustomTitle',
#             parent=styles['Heading1'],
#             fontSize=24,
#             spaceAfter=30,
#             textColor=colors.darkblue,
#             alignment=1  # Center alignment
#         )
        
#         subtitle_style = ParagraphStyle(
#             'CustomSubtitle',
#             parent=styles['Heading2'],
#             fontSize=16,
#             spaceAfter=20,
#             textColor=colors.darkgreen
#         )
        
#         # Title and Header
#         story.append(Paragraph("🚀 NexGen AI Resume Analysis Report", title_style))
#         story.append(Spacer(1, 20))
        
#         # Report Info Table
#         report_data = [
#             ['Report Generated:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
#             ['Candidate Name:', candidate_name or 'N/A'],
#             ['Position Applied:', job_title or 'N/A'],
#             ['Analysis Engine:', 'NexGen AI v2.0']
#         ]
        
#         info_table = Table(report_data, colWidths=[2*inch, 4*inch])
#         info_table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
#             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
#             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, -1), 10),
#             ('GRID', (0, 0), (-1, -1), 1, colors.black),
#             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#             ('TOPPADDING', (0, 0), (-1, -1), 8),
#             ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
#         ]))
#         story.append(info_table)
#         story.append(Spacer(1, 30))
        
#         # Executive Summary
#         story.append(Paragraph("📊 Executive Summary", subtitle_style))
#         summary_text = f"""
#         The AI analysis reveals a <b>{analysis['overall_match_score']:.1f}%</b> overall match score for this position.
#         The candidate demonstrates strong alignment in <b>{len(analysis['matching_skills'])}</b> key skill areas,
#         with <b>{len(analysis['missing_skills'])}</b> areas identified for development.
#         """
#         story.append(Paragraph(summary_text, styles['Normal']))
#         story.append(Spacer(1, 20))
        
#         # Score Breakdown Table
#         story.append(Paragraph("🎯 Score Breakdown", subtitle_style))
#         score_data = [
#             ['Metric', 'Score', 'Interpretation'],
#             ['Overall Match', f"{analysis['overall_match_score']:.1f}%", 
#              'Excellent' if analysis['overall_match_score'] >= 80 else 
#              'Good' if analysis['overall_match_score'] >= 60 else 'Needs Improvement'],
#             ['Semantic Score', f"{analysis['semantic_score']:.1f}%", 
#              'High contextual alignment' if analysis['semantic_score'] >= 70 else 'Moderate alignment'],
#             ['Skill Score', f"{analysis['skill_score']:.1f}%", 
#              'Strong skill match' if analysis['skill_score'] >= 70 else 'Skill gap identified']
#         ]
        
#         score_table = Table(score_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
#         score_table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, 0), 12),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#             ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
#             ('GRID', (0, 0), (-1, -1), 1, colors.black),
#             ('FONTSIZE', (0, 1), (-1, -1), 10),
#             ('TOPPADDING', (0, 1), (-1, -1), 8),
#             ('BOTTOMPADDING', (0, 1), (-1, -1), 8)
#         ]))
#         story.append(score_table)
#         story.append(Spacer(1, 25))
        
#         # Skills Analysis
#         story.append(Paragraph("🔧 Skills Analysis", subtitle_style))
        
#         # Matching Skills
#         matching_skills_text = "<b>✅ Matching Skills:</b><br/>" + ", ".join(analysis['matching_skills'][:15])
#         if len(analysis['matching_skills']) > 15:
#             matching_skills_text += f"<br/><i>...and {len(analysis['matching_skills']) - 15} more</i>"
#         story.append(Paragraph(matching_skills_text, styles['Normal']))
#         story.append(Spacer(1, 15))
        
#         # Missing Skills
#         missing_skills_text = "<b>🎯 Skills to Develop:</b><br/>" + ", ".join(analysis['missing_skills'][:10])
#         if len(analysis['missing_skills']) > 10:
#             missing_skills_text += f"<br/><i>...and {len(analysis['missing_skills']) - 10} more</i>"
#         story.append(Paragraph(missing_skills_text, styles['Normal']))
#         story.append(Spacer(1, 25))
        
#         # AI Insights
#         if analysis.get('ai_insights'):
#             story.append(Paragraph("🤖 AI-Powered Insights", subtitle_style))
#             insights = analysis['ai_insights']
            
#             # LLM Analysis if available
#             if insights.get('llm_analysis'):
#                 story.append(Paragraph("<b>Advanced AI Analysis:</b>", styles['Normal']))
#                 story.append(Paragraph(insights['llm_analysis'], styles['Normal']))
#                 story.append(Spacer(1, 15))
            
#             # Strengths
#             if insights.get('strengths'):
#                 strengths_text = "<b>💪 Identified Strengths:</b><br/>" + "<br/>".join([f"• {s}" for s in insights['strengths']])
#                 story.append(Paragraph(strengths_text, styles['Normal']))
#                 story.append(Spacer(1, 15))
            
#             # Recommendations
#             if insights.get('recommendations'):
#                 recs_text = "<b>📈 Recommendations:</b><br/>" + "<br/>".join([f"• {r}" for r in insights['recommendations']])
#                 story.append(Paragraph(recs_text, styles['Normal']))
#                 story.append(Spacer(1, 25))
        
#         # Enhancement Suggestions
#         if analysis.get('enhancement_suggestions'):
#             story.append(Paragraph("🚀 Career Enhancement Plan", subtitle_style))
#             for i, suggestion in enumerate(analysis['enhancement_suggestions'][:5], 1):
#                 suggestion_text = f"<b>{i}. {suggestion['skill']} ({suggestion['priority']} Priority)</b><br/>{suggestion['action']}"
#                 story.append(Paragraph(suggestion_text, styles['Normal']))
#                 story.append(Spacer(1, 10))
        
#         # Footer
#         story.append(Spacer(1, 30))
#         footer_text = """
#         <i>This report was generated by NexGen AI Resume Analyzer - Advanced Career Intelligence Platform.<br/>
#         For more information, visit our platform or contact support.</i>
#         """
#         story.append(Paragraph(footer_text, styles['Normal']))
        
#         # Build PDF
#         doc.build(story)
#         buffer.seek(0)
#         return buffer.getvalue()
        
#     except Exception as e:
#         st.error(f"PDF generation error: {e}")
#         return b""

# # =============================================================================
# # VISUALIZATIONS
# # =============================================================================

# def create_visualizations(analysis_data: Dict[str, Any]):
#     """Create interactive visualizations"""
    
#     # Match Score Gauge
#     fig_gauge = go.Figure(go.Indicator(
#         mode = "gauge+number+delta",
#         value = analysis_data["overall_match_score"],
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         title = {'text': "AI Match Score"},
#         delta = {'reference': 80},
#         gauge = {
#             'axis': {'range': [None, 100]},
#             'bar': {'color': "darkblue"},
#             'steps': [
#                 {'range': [0, 50], 'color': "lightgray"},
#                 {'range': [50, 80], 'color': "yellow"},
#                 {'range': [80, 100], 'color': "green"}
#             ]
#         }
#     ))
#     fig_gauge.update_layout(height=300)
    
#     # Skills Chart
#     matching_count = len(analysis_data["matching_skills"])
#     missing_count = len(analysis_data["missing_skills"])
    
#     fig_pie = go.Figure(data=[go.Pie(
#         labels=['Matching Skills', 'Missing Skills'],
#         values=[matching_count, missing_count],
#         hole=.5,
#         marker_colors=['#00d4aa', '#ff6b6b']
#     )])
#     fig_pie.update_layout(title="Skills Analysis", height=300)
    
#     return fig_gauge, fig_pie

# # =============================================================================
# # SESSION MANAGEMENT
# # =============================================================================

# def init_session_state():
#     """Initialize session state with all required variables"""
#     if 'authenticated' not in st.session_state:
#         st.session_state.authenticated = False
#     if 'user_info' not in st.session_state:
#         st.session_state.user_info = None
#     if 'current_analysis' not in st.session_state:
#         st.session_state.current_analysis = None
#     if 'candidate_name' not in st.session_state:
#         st.session_state.candidate_name = ""
#     if 'job_title' not in st.session_state:
#         st.session_state.job_title = ""
#     if 'pdf_generated' not in st.session_state:
#         st.session_state.pdf_generated = False
#     if 'pdf_data' not in st.session_state:
#         st.session_state.pdf_data = None
#     if 'jobs_searched' not in st.session_state:
#         st.session_state.jobs_searched = False
#     if 'job_results' not in st.session_state:
#         st.session_state.job_results = []

# def render_auth_sidebar():
#     """Authentication sidebar"""
#     with st.sidebar:
#         if st.session_state.authenticated:
#             user = st.session_state.user_info
#             st.success(f"✅ Welcome, **{user['username']}**!")
            
#             if st.button("🚪 Logout", use_container_width=True):
#                 st.session_state.authenticated = False
#                 st.session_state.user_info = None
#                 st.success("Logged out successfully!")
#                 st.rerun()
        
#         else:
#             st.header("🔐 Access NexGen AI")
            
#             tab = st.selectbox("", ["Login", "Sign Up"])
            
#             if tab == "Login":
#                 with st.form("login"):
#                     username = st.text_input("Username/Email")
#                     password = st.text_input("Password", type="password")
                    
#                     if st.form_submit_button("Login", use_container_width=True):
#                         if username and password:
#                             user = NexGenAuth.authenticate(username.strip(), password)
#                             if user:
#                                 st.session_state.authenticated = True
#                                 st.session_state.user_info = user
#                                 st.success("🎉 Login successful!")
#                                 st.rerun()
#                             else:
#                                 st.error("❌ Invalid credentials")
            
#             else:
#                 with st.form("signup"):
#                     username = st.text_input("Username")
#                     email = st.text_input("Email")
#                     full_name = st.text_input("Full Name")
#                     password = st.text_input("Password", type="password")
#                     confirm = st.text_input("Confirm Password", type="password")
                    
#                     if st.form_submit_button("Create Account", use_container_width=True):
#                         if password != confirm:
#                             st.error("Passwords don't match")
#                         else:
#                             success, msg = NexGenAuth.create_user(username, email, password, full_name)
#                             if success:
#                                 st.success("Account created! Please login.")
#                             else:
#                                 st.error(msg)

# # =============================================================================
# # MAIN INTERFACE
# # =============================================================================

# def render_main_analyzer():
#     """Main analysis interface"""
#     st.markdown('<div class="nexgen-header"><h1>🚀 NexGen AI Resume Analyzer</h1><p>Advanced AI-Powered Career Intelligence Platform</p></div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📄 Upload Resume")
#         uploaded_file = st.file_uploader("Choose file", type=['pdf', 'docx', 'txt'])
#         candidate_name = st.text_input("Candidate Name")
    
#     with col2:
#         st.subheader("💼 Job Description")
#         job_description = st.text_area("Paste job description", height=200)
#         job_title = st.text_input("Job Title")
    
#     if st.button("🔍 Analyze with NexGen AI", type="primary", use_container_width=True):
#         if uploaded_file and job_description:
#             with st.spinner("🤖 NexGen AI is analyzing..."):
#                 resume_text = extract_text_from_file(uploaded_file)
                
#                 if resume_text:
#                     analysis = nexgen_ai.analyze_resume_job_match(resume_text, job_description)
                    
#                     # Store in session state
#                     st.session_state.current_analysis = analysis
#                     st.session_state.candidate_name = candidate_name
#                     st.session_state.job_title = job_title
#                     st.session_state.pdf_generated = False  # Reset PDF flag
#                     st.session_state.jobs_searched = False  # Reset jobs flag
                    
#                     display_analysis_results(analysis, candidate_name, job_title)
#                 else:
#                     st.error("Could not extract text from file")
#         else:
#             st.error("Please upload resume and provide job description")

# def display_analysis_results(analysis: Dict[str, Any], candidate_name: str, job_title: str):
#     """Display analysis results"""
#     st.markdown("---")
#     st.header("📊 NexGen AI Analysis Results")
    
#     # Metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.markdown(f'<div class="ai-metric"><h3>{analysis["overall_match_score"]:.1f}%</h3><p>Overall Match</p></div>', unsafe_allow_html=True)
    
#     with col2:
#         st.markdown(f'<div class="ai-metric"><h3>{analysis["semantic_score"]:.1f}%</h3><p>AI Semantic Score</p></div>', unsafe_allow_html=True)
    
#     with col3:
#         st.markdown(f'<div class="ai-metric"><h3>{len(analysis["matching_skills"])}</h3><p>Matching Skills</p></div>', unsafe_allow_html=True)
    
#     with col4:
#         st.markdown(f'<div class="ai-metric"><h3>{len(analysis["missing_skills"])}</h3><p>Skills to Develop</p></div>', unsafe_allow_html=True)
    
#     # Visualizations
#     st.subheader("📈 Visual Intelligence")
#     fig_gauge, fig_pie = create_visualizations(analysis)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_gauge, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_pie, use_container_width=True)
    
#     # Skills Analysis
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("✅ Matching Skills")
#         for skill in analysis["matching_skills"]:
#             st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
#     with col2:
#         st.subheader("🎯 Skills to Develop")
#         for skill in analysis["missing_skills"][:10]:
#             st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
    
#     # AI Insights
#     if analysis.get("ai_insights"):
#         st.subheader("🤖 AI-Powered Insights")
#         insights = analysis["ai_insights"]
        
#         # LLM Analysis if available
#         if insights.get('llm_analysis'):
#             st.markdown("**🧠 Advanced LLM Analysis:**")
#             st.info(insights['llm_analysis'])
#             st.markdown("---")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("**💪 Strengths Identified:**")
#             for strength in insights.get("strengths", []):
#                 st.write(f"• {strength}")
        
#         with col2:
#             st.write("**📈 AI Recommendations:**")
#             for rec in insights.get("recommendations", []):
#                 st.write(f"• {rec}")
        
#         # Sentiment Analysis
#         if insights.get('sentiment'):
#             sentiment = insights['sentiment']
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("💭 Sentiment Score", f"{sentiment.get('polarity', 0):.2f}")
#             with col2:
#                 st.metric("🗣️ Communication Tone", sentiment.get('tone', 'Professional'))
    
#     # Enhancement Suggestions
#     if analysis.get("enhancement_suggestions"):
#         st.subheader("🚀 Career Enhancement Plan")
#         for suggestion in analysis["enhancement_suggestions"]:
#             with st.expander(f"📈 {suggestion['skill']} - {suggestion['priority']} Priority"):
#                 st.write(suggestion["action"])
    
#     # =============================================================================
#     # PDF REPORT GENERATION - BULLETPROOF IMPLEMENTATION
#     # =============================================================================
    
#     st.markdown("---")
#     st.subheader("📄 Professional PDF Report Generator")
    
#     # Check if we have analysis data
#     if not analysis or not analysis.get('overall_match_score'):
#         st.warning("⚠️ Complete the resume analysis above first to generate PDF report")
#         return
    
#     # PDF Generation UI
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         st.markdown("**🎆 Generate comprehensive PDF report with all analysis results**")
        
#         # PDF generation button with unique key
#         if st.button(
#             "📄 🎆 CREATE PDF REPORT NOW",
#             type="primary",
#             use_container_width=True,
#             key="pdf_gen_button",
#             help="Click to generate and download professional PDF report"
#         ):
#             # Force generate PDF regardless of previous state
#             try:
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 status_text.text("🔄 Initializing PDF generator...")
#                 progress_bar.progress(20)
                
#                 # Try to generate PDF
#                 status_text.text("📝 Creating report content...")
#                 progress_bar.progress(40)
                
#                 pdf_bytes = generate_pdf_report(
#                     analysis, 
#                     candidate_name or "Job Candidate", 
#                     job_title or "Position Applied"
#                 )
                
#                 progress_bar.progress(70)
#                 status_text.text("🔌 Finalizing document...")
                
#                 if pdf_bytes and len(pdf_bytes) > 0:
#                     progress_bar.progress(100)
#                     status_text.text("✅ PDF Generated Successfully!")
                    
#                     # Store PDF in session state
#                     st.session_state.pdf_data = pdf_bytes
#                     st.session_state.pdf_generated = True
                    
#                     # Create download filename
#                     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#                     filename = f"NexGen_Resume_Analysis_{candidate_name or 'Report'}_{timestamp}.pdf"
                    
#                     # SUCCESS - Show download
#                     st.success("🎉 ✅ **PDF REPORT CREATED SUCCESSFULLY!**")
#                     st.balloons()
                    
#                     # Download button
#                     st.download_button(
#                         label=f"📥 🎆 DOWNLOAD PDF REPORT ({len(pdf_bytes)} bytes)",
#                         data=pdf_bytes,
#                         file_name=filename,
#                         mime="application/pdf",
#                         type="primary",
#                         use_container_width=True,
#                         key="pdf_download_button"
#                     )
                    
#                     # Clear progress
#                     progress_bar.empty()
#                     status_text.empty()
                    
#                 else:
#                     raise Exception("PDF generation returned empty data")
                    
#             except ImportError as e:
#                 st.error("🚨 **MISSING DEPENDENCY:** ReportLab not installed")
#                 st.code("pip install reportlab", language="bash")
#                 st.info("💡 **Quick Fix:** Run the command above in your terminal, then refresh this page")
                
#             except Exception as e:
#                 st.error(f"🚨 **PDF Generation Failed:** {str(e)}")
                
#                 # Provide fallback text report
#                 st.warning("📄 **Alternative: Text Report**")
                
#                 text_content = f"""NEXGEN AI RESUME ANALYSIS REPORT
# {'='*50}

# CANDIDATE: {candidate_name or 'N/A'}
# POSITION: {job_title or 'N/A'}
# DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

# OVERALL MATCH SCORE: {analysis['overall_match_score']:.1f}%
# SEMANTIC SCORE: {analysis['semantic_score']:.1f}%
# SKILL SCORE: {analysis['skill_score']:.1f}%

# MATCHING SKILLS ({len(analysis['matching_skills'])}):
# {', '.join(analysis['matching_skills']) if analysis['matching_skills'] else 'None found'}

# MISSING SKILLS ({len(analysis['missing_skills'])}):
# {', '.join(analysis['missing_skills']) if analysis['missing_skills'] else 'None identified'}

# AI INSIGHTS:
# {analysis.get('ai_insights', {}).get('llm_analysis', 'Analysis completed successfully')}

# RECOMMENDATIONS:
# {chr(10).join([f'- {rec}' for rec in analysis.get('ai_insights', {}).get('recommendations', ['Complete analysis for detailed recommendations'])])}

# Generated by NexGen AI Resume Analyzer
# """
                
#                 st.download_button(
#                     label="📄 Download Text Report",
#                     data=text_content,
#                     file_name=f"resume_analysis_text_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
#                     mime="text/plain",
#                     use_container_width=True
#                 )
    
#     with col2:
#         st.info("📈 **Report Features:**\n\n• Executive Summary\n• Score Breakdown\n• Skills Matrix\n• AI Insights\n• Career Plan\n• Professional Layout")
        
#         # Show PDF status
#         if st.session_state.get('pdf_generated', False):
#             st.success("✅ **PDF Ready**\nClick download above")
    
#     # =============================================================================
#     # GLOBAL JOB RECOMMENDATIONS - BULLETPROOF IMPLEMENTATION  
#     # =============================================================================
    
#     st.markdown("---")
#     st.subheader("🌍 AI-Powered Global Job Finder")
    
#     # Check if we have analysis data
#     if not analysis or not analysis.get('matching_skills'):
#         st.warning("⚠️ Complete the resume analysis above first to get personalized job recommendations")
#         return
    
#     # Country and location selection
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         country_map = {
#             "🇺🇸 United States": "us",
#             "🇬🇧 United Kingdom": "gb", 
#             "🇨🇦 Canada": "ca",
#             "🇦🇺 Australia": "au",
#             "🇩🇪 Germany": "de",
#             "🇫🇷 France": "fr",
#             "🇳🇱 Netherlands": "nl",
#             "🇮🇳 India": "in",
#             "🇸🇬 Singapore": "sg",
#             "🇧🇷 Brazil": "br",
#             "🇯🇵 Japan": "jp",
#             "🇪🇸 Spain": "es"
#         }
        
#         selected_country_display = st.selectbox(
#             "🌍 **Select Target Country**", 
#             options=list(country_map.keys()),
#             index=0,
#             key="country_selector",
#             help="Choose where you want to find job opportunities"
#         )
#         country_code = country_map[selected_country_display]
#         country_name = selected_country_display.split(' ', 1)[1]  # Remove flag
    
#     with col2:
#         city_input = st.text_input(
#             "🏢 **City/Region**", 
#             placeholder="e.g., London, Toronto, Sydney",
#             key="city_input",
#             help="Specify city for targeted results (optional)"
#         )
    
#     with col3:
#         include_remote = st.checkbox(
#             "🏠 **Include Remote Work**", 
#             value=True,
#             key="remote_checkbox",
#             help="Include work-from-home opportunities"
#         )
    
#     # Display user's skills for context
#     st.markdown(f"**🎯 Your Matching Skills ({len(analysis['matching_skills'])}):**")
#     skills_text = ", ".join(analysis['matching_skills'][:10])
#     if len(analysis['matching_skills']) > 10:
#         skills_text += f" +{len(analysis['matching_skills']) - 10} more"
#     st.info(skills_text)
    
#     # Job Search Button - BULLETPROOF
#     if st.button(
#         f"🚀 FIND JOBS IN {country_name.upper()}",
#         type="primary",
#         use_container_width=True,
#         key="job_search_button",
#         help=f"Search for jobs matching your skills in {country_name}"
#     ):
#         # FORCE JOB SEARCH - NO CONDITIONS
#         try:
#             # Progress tracking
#             progress_container = st.container()
#             with progress_container:
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 # Step 1: Initialize
#                 status_text.text("🔍 Initializing global job search...")
#                 progress_bar.progress(10)
                
#                 # Step 2: Search primary jobs
#                 status_text.text(f"🌍 Searching jobs in {country_name}...")
#                 progress_bar.progress(30)
                
#                 all_jobs = []
                
#                 # Always get sample jobs first (guaranteed to work)
#                 sample_jobs = job_engine._get_sample_jobs(analysis['matching_skills'], country_code)
#                 all_jobs.extend(sample_jobs)
                
#                 progress_bar.progress(50)
#                 status_text.text("🏠 Adding remote opportunities...")
                
#                 # Add remote jobs if requested
#                 if include_remote:
#                     try:
#                         remote_jobs = job_engine.search_jobs_remotive(analysis['matching_skills'], max_results=3)
#                         all_jobs.extend(remote_jobs)
#                     except:
#                         pass  # Don't break if remote search fails
                
#                 progress_bar.progress(70)
#                 status_text.text("📈 Ranking opportunities by match score...")
                
#                 # Try to get real API jobs (but don't break if fails)
#                 try:
#                     api_jobs = job_engine.search_jobs_adzuna(
#                         analysis['matching_skills'], 
#                         country_code, 
#                         city_input,
#                         max_results=5
#                     )
#                     # Only add if we got real results (not just samples)
#                     if api_jobs and len(api_jobs) > 0:
#                         # Replace sample jobs with real ones if available
#                         all_jobs = api_jobs + [job for job in all_jobs if job.get('url') != 'https://example.com/job1']
#                 except:
#                     pass  # Keep sample jobs if API fails
                
#                 progress_bar.progress(90)
#                 status_text.text("✅ Finalizing results...")
                
#                 # Sort by match score
#                 all_jobs = sorted(
#                     all_jobs, 
#                     key=lambda x: x.get('match_score', 0), 
#                     reverse=True
#                 )
                
#                 progress_bar.progress(100)
#                 status_text.text(f"✅ Found {len(all_jobs)} opportunities!")
                
#                 # Store results in session
#                 st.session_state.job_results = all_jobs
#                 st.session_state.jobs_searched = True
                
#                 # Clear progress
#                 progress_bar.empty()
#                 status_text.empty()
            
#             # ALWAYS SHOW RESULTS (even if just samples)
#             if all_jobs:
#                 # Success message
#                 st.success(f"🎉 **FOUND {len(all_jobs)} JOB OPPORTUNITIES IN {country_name.upper()}!**")
                
#                 # Job statistics
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     avg_score = sum(job.get('match_score', 0) for job in all_jobs) / len(all_jobs)
#                     st.metric("🎯 Avg Match", f"{avg_score:.0f}%")
#                 with col2:
#                     best_score = max(job.get('match_score', 0) for job in all_jobs)
#                     st.metric("🏆 Best Match", f"{best_score:.0f}%")
#                 with col3:
#                     remote_count = len([j for j in all_jobs if 'remote' in j.get('location', '').lower()])
#                     st.metric("🏠 Remote Jobs", remote_count)
#                 with col4:
#                     st.metric("💼 Total Jobs", len(all_jobs))
                
#                 st.markdown("---")
                
#                 # Display jobs in cards
#                 st.markdown(f"### 💼 **Top Opportunities in {country_name}**")
                
#                 for i, job in enumerate(all_jobs[:12], 1):
#                     # Determine match level
#                     match_score = job.get('match_score', 0)
#                     if match_score >= 80:
#                         match_icon = "🏆"
#                         match_color = "success"
#                     elif match_score >= 60:
#                         match_icon = "🎯"
#                         match_color = "warning"
#                     else:
#                         match_icon = "💼"
#                         match_color = "info"
                    
#                     # Job card
#                     with st.expander(
#                         f"{match_icon} **{job['title']}** @ **{job['company']}** - {match_score:.0f}% Match",
#                         expanded=(i <= 3)  # Show top 3 expanded
#                     ):
#                         col1, col2 = st.columns([3, 1])
                        
#                         with col1:
#                             st.markdown(f"**🏢 Company:** {job['company']}")
#                             st.markdown(f"**📍 Location:** {job['location']}")
#                             st.markdown(f"**💰 Salary:** {job['salary']}")
#                             st.markdown(f"**📅 Posted:** {job.get('created', 'Recently')}")
                            
#                             # Description
#                             st.markdown("**📋 Description:**")
#                             st.write(job.get('description', 'Exciting opportunity in a growing company.'))
                            
#                             # Skills match indicator
#                             job_desc_lower = job.get('description', '').lower()
#                             matching_user_skills = [
#                                 skill for skill in analysis['matching_skills']
#                                 if skill.lower() in job_desc_lower
#                             ]
#                             if matching_user_skills:
#                                 st.markdown(f"**✅ Your matching skills:** {', '.join(matching_user_skills[:5])}")
                        
#                         with col2:
#                             # Match score with appropriate styling
#                             if match_color == "success":
#                                 st.success(f"🏆 {match_score:.0f}% Match")
#                             elif match_color == "warning":
#                                 st.warning(f"🎯 {match_score:.0f}% Match")
#                             else:
#                                 st.info(f"💼 {match_score:.0f}% Match")
                            
#                             # Apply button
#                             job_url = job.get('url', '')
#                             if job_url and job_url != '#' and not job_url.startswith('https://example.com'):
#                                 st.link_button(
#                                     "🔗 APPLY NOW", 
#                                     job_url,
#                                     use_container_width=True
#                                 )
#                             else:
#                                 st.info("📝 Sample Job\n(Real jobs available\nwith API setup)")
                            
#                             # Country indicator
#                             country_flags = {
#                                 'us': '🇺🇸', 'gb': '🇬🇧', 'ca': '🇨🇦',
#                                 'au': '🇦🇺', 'de': '🇩🇪', 'fr': '🇫🇷',
#                                 'nl': '🇳🇱', 'in': '🇮🇳', 'sg': '🇸🇬'
#                             }
#                             flag = country_flags.get(country_code, '🌍')
#                             st.markdown(f"**{flag} {country_name}**")
                
#                 # Additional tips
#                 st.markdown("### 💡 **Job Search Tips**")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.info("🎆 **To get REAL job results:**\n- Set up Adzuna API key\n- Configure OpenAI for better matching\n- See SETUP_GUIDE.md for details")
#                 with col2:
#                     st.success("🚀 **Boost your chances:**\n- Improve missing skills\n- Customize resume for each job\n- Apply within 48 hours of posting")
            
#             else:
#                 # This should never happen with our bulletproof approach
#                 st.error("🚨 Unexpected error: No jobs found")
#                 st.info("💡 Please try refreshing the page or contact support")
        
#         except Exception as e:
#             st.error(f"🚨 **Job Search Error:** {str(e)}")
            
#             # Always provide fallback
#             st.warning("🔄 **Showing sample opportunities as fallback:**")
#             fallback_jobs = job_engine._get_sample_jobs(analysis['matching_skills'], country_code)
            
#             for job in fallback_jobs[:3]:
#                 with st.expander(f"💼 {job['title']} @ {job['company']}"):
#                     st.write(f"**Location:** {job['location']}")
#                     st.write(f"**Salary:** {job['salary']}")
#                     st.write(f"**Match:** {job.get('match_score', 0):.0f}%")
#                     st.write(f"**Description:** {job['description']}")

# # =============================================================================
# # MAIN APPLICATION
# # =============================================================================

# def main():
#     """Main application"""
#     init_session_state()
#     render_auth_sidebar()
    
#     if st.session_state.authenticated:
#         render_main_analyzer()
#     else:
#         st.markdown("""
#         <div class="nexgen-header">
#             <h1>🚀 Welcome to NexGen AI</h1>
#             <p>The Future of AI-Powered Career Growth</p>
#             <p>Please login or create an account to access the world's most advanced resume analyzer.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Demo features for non-authenticated users
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.markdown("""
#             ### 🤖 AI-Powered Analysis
#             - Semantic matching using advanced NLP
#             - Skill extraction and categorization
#             - Sentiment analysis of resume content
#             """)
        
#         with col2:
#             st.markdown("""
#             ### 📊 Professional Reports
#             - Interactive visualizations
#             - PDF report generation
#             - Detailed enhancement suggestions
#             """)
        
#         with col3:
#             st.markdown("""
#             ### 🚀 Career Growth
#             - Personalized recommendations
#             - Skills gap analysis
#             - Industry insights
#             """)

# if __name__ == "__main__":
#     main()
