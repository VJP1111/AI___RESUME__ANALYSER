# =========================
# World's Best Resume Analyzer (Streamlit)
# Auth (SQLite) + Advanced NLP + Visuals + PDF Report
# =========================

import os
import re
import io
import base64
import sqlite3
import hashlib
import tempfile
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# NLP & Similarity
import spacy
from sentence_transformers import SentenceTransformer, util

# File parsing
import PyPDF2
try:
    import docx  # python-docx (optional)
except Exception:
    docx = None

# Visuals
import plotly.graph_objects as go

# PDF
from fpdf import FPDF


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="World-Class Resume Analyzer", page_icon="🧠", layout="wide")

APP_DB = "users.db"  # SQLite file
MODEL_NAME = "all-MiniLM-L6-v2"

# A curated (but compact) skills lexicon. You can expand freely.
SKILLS = [
    # Programming
    "Python","Java","JavaScript","TypeScript","C","C++","C#","Go","Rust","R","MATLAB","Scala","Kotlin","PHP",
    # Data / ML / AI
    "SQL","NoSQL","MySQL","PostgreSQL","MongoDB","Redis","Elasticsearch","Pandas","NumPy","SciPy","Scikit-learn",
    "TensorFlow","Keras","PyTorch","XGBoost","LightGBM","Machine Learning","Deep Learning","NLP","Computer Vision",
    "LLM","Prompt Engineering","MLOps","Data Engineering","Data Analysis","Statistics",
    # Web / Backend / Cloud
    "HTML","CSS","React","Next.js","Node.js","Express","Django","Flask","FastAPI","GraphQL","REST API",
    "AWS","Azure","GCP","Docker","Kubernetes","CI/CD","Jenkins","GitHub Actions","Terraform","Linux","Git",
    # Project / PM / Soft
    "Agile","Scrum","Jira","Confluence","Project Management","Communication","Teamwork","Leadership","Problem Solving"
]

MULTIWORD_SKILLS = [
    "Machine Learning","Deep Learning","Natural Language Processing","Computer Vision","Data Science",
    "Data Engineering","Prompt Engineering","Large Language Models","Time Series Analysis","Distributed Systems",
    "Object Oriented Programming","Continuous Integration","Continuous Delivery","Microservices",
    "Cloud Computing","Test Driven Development","Feature Engineering","Hyperparameter Tuning"
]


# -----------------------------
# AUTH (SQLite + salted SHA256)
# -----------------------------
def init_db():
    con = sqlite3.connect(APP_DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_user(username: str, password: str) -> bool:
    try:
        salt = base64.b16encode(os.urandom(16)).decode("ascii")
        pw_hash = _hash_pw(password, salt)
        con = sqlite3.connect(APP_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO users (username, salt, password_hash, created_at) VALUES (?,?,?,?)",
                    (username, salt, pw_hash, datetime.utcnow().isoformat()))
        con.commit()
        con.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    con = sqlite3.connect(APP_DB)
    cur = con.cursor()
    cur.execute("SELECT salt, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    salt, pw_hash = row
    return _hash_pw(password, salt) == pw_hash


# -----------------------------
# NLP MODELS (robust loading)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model not found. Please run: `python -m spacy download en_core_web_sm`")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_st_model():
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        st.error("Failed to load SentenceTransformer. Install 'sentence-transformers' and 'torch'.")
        st.stop()


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(file) -> str:
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def extract_text_from_docx(file) -> str:
    if docx is None:
        return ""
    try:
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def read_resume_file(uploaded) -> str:
    if uploaded.type == "application/pdf":
        return extract_text_from_pdf(uploaded)
    if uploaded.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        return extract_text_from_docx(uploaded)
    # Fallback to plain text
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


# -----------------------------
# SKILL & ENTITY EXTRACTION
# -----------------------------
def normalize_tokens(tokens):
    return [t.strip().lower() for t in tokens if t and t.strip()]

def dedup_preserve_order(seq):
    seen = set()
    out = []
    for s in seq:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def extract_candidate_name(text: str, nlp) -> str:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # fallback: first capitalized token cluster
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    return m.group(1) if m else "Candidate"

def extract_job_title(text: str, nlp) -> str:
    # Try heuristic: look for 'for <Title>' or 'Position: <Title>'
    m = re.search(r"(?:for|position|role)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 /\-&]+)", text, flags=re.I)
    if m:
        candidate = m.group(1).split("\n")[0].strip()
        return candidate[:60]
    # Try NER ORG/PRODUCT/WORK_OF_ART as fallback (often job titles aren't tagged; grab early noun chunk)
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        if chunk.text.istitle() and 2 <= len(chunk.text.split()) <= 5:
            return chunk.text
    return "Job Title"

def get_skill_candidates(text: str, nlp) -> list:
    text_lower = text.lower()
    hits = []

    # Exact dictionary hits (case-insensitive)
    for s in SKILLS + MULTIWORD_SKILLS:
        if s.lower() in text_lower:
            hits.append(s)

    # Add noun chunks as potential skills
    doc = nlp(text)
    for nc in doc.noun_chunks:
        token = nc.text.strip()
        if 2 <= len(token) <= 40 and any(c.isalpha() for c in token):
            # keep multiword tech-y chunks
            if token.lower() not in normalize_tokens(hits):
                hits.append(token)

    return dedup_preserve_order(hits)


# -----------------------------
# MATCHING & SCORING
# -----------------------------
def semantic_skill_match(resume_skills, jd_skills, model):
    """
    Returns:
      exact_matches (set),
      mapping list of dicts for each jd skill: {jd_skill, exact, closest_resume_skill, similarity}
      score_percent (float)
    """
    jd_list = dedup_preserve_order(jd_skills)
    res_list = dedup_preserve_order(resume_skills)

    exact = set()
    # Normalize for exact matching
    jd_norm = [s.lower() for s in jd_list]
    res_norm = [s.lower() for s in res_list]

    # Exact matches by normalization
    for j, jn in zip(jd_list, jd_norm):
        if jn in res_norm:
            exact.add(j)

    # Semantic matching
    if res_list and jd_list:
        emb_jd = model.encode(jd_list, convert_to_tensor=True, normalize_embeddings=True)
        emb_res = model.encode(res_list, convert_to_tensor=True, normalize_embeddings=True)
        sim = util.cos_sim(emb_jd, emb_res).cpu().numpy()  # shape (len(jd), len(res))

        mapping = []
        best_per_jd = sim.max(axis=1) if sim.size else np.zeros(len(jd_list))
        best_idx = sim.argmax(axis=1) if sim.size else np.zeros(len(jd_list), dtype=int)

        for i, jd_skill in enumerate(jd_list):
            closest = res_list[int(best_idx[i])] if len(res_list) else ""
            similarity = float(best_per_jd[i])
            mapping.append({
                "jd_skill": jd_skill,
                "exact": jd_skill in exact,
                "closest_resume_skill": closest,
                "similarity": similarity
            })

        # Final score: combine coverage and semantic mean
        coverage = len(exact) / max(1, len(jd_list))
        semantic_mean = float(best_per_jd.mean()) if len(best_per_jd) else 0.0
        score = 100.0 * (0.5 * coverage + 0.5 * semantic_mean)
    else:
        mapping = []
        score = 0.0

    return exact, mapping, round(score, 2)

def rank_missing(mapping):
    rows = []
    for m in mapping:
        if not m["exact"]:
            sim = m["similarity"]
            # Priority heuristic
            if sim >= 0.7:
                pr = "Low"
            elif sim >= 0.4:
                pr = "Medium"
            else:
                pr = "High"
            rows.append({
                "Required Skill": m["jd_skill"],
                "Closest Resume Skill": m["closest_resume_skill"],
                "Similarity": round(sim * 100, 1),
                "Priority": pr
            })
    # High -> Medium -> Low, then by similarity ascending
    order = {"High": 0, "Medium": 1, "Low": 2}
    rows.sort(key=lambda x: (order[x["Priority"]], x["Similarity"]))
    return pd.DataFrame(rows)


# -----------------------------
# VISUALS (NO BAR CHARTS)
# -----------------------------
def plot_gauge(score_percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_percent,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, 40]},
                {"range": [40, 70]},
                {"range": [70, 100]}
            ]
        },
        title={"text": "Overall Match"}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_donut(matched_count: int, missing_count: int):
    fig = go.Figure(go.Pie(
        labels=["Matching Skills", "Missing Skills"],
        values=[matched_count, missing_count],
        hole=0.5
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# -----------------------------
# PDF REPORT
# -----------------------------
def create_pdf(candidate_name, job_title, username, score, donut_path, gauge_path,
               matching_list, missing_df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Resume Analysis Report", 0, 1, 'C')
    pdf.ln(2)

    # Meta
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Candidate: {candidate_name}", 0, 1)
    pdf.cell(0, 8, f"Job Title: {job_title}", 0, 1)
    pdf.cell(0, 8, f"Generated by: {username}", 0, 1)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f"Overall Match Score: {score:.2f}%", 0, 1)

    # Images
    if gauge_path:
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, "Overall Match Gauge:", 0, 1)
        pdf.image(gauge_path, x=20, y=None, w=170)

    if donut_path:
        pdf.ln(4)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, "Matching vs Missing (Donut):", 0, 1)
        pdf.image(donut_path, x=20, y=None, w=170)

    # Matching list
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Matching Skills:", 0, 1)
    pdf.set_font("Arial", '', 10)
    if matching_list:
        match_str = ", ".join(sorted(matching_list))
        pdf.multi_cell(0, 5, match_str)
    else:
        pdf.multi_cell(0, 5, "No exact matching skills.")

    # Missing + Suggestions
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Top Enhancement Suggestions:", 0, 1)
    pdf.set_font("Arial", '', 10)
    if missing_df is not None and not missing_df.empty:
        top_rows = missing_df.head(10).to_dict(orient="records")
        for r in top_rows:
            rec = suggestion_for_skill(r["Required Skill"], r["Closest Resume Skill"], r["Similarity"])
            pdf.multi_cell(0, 5, f"- {rec}")
    else:
        pdf.multi_cell(0, 5, "No missing skills. Excellent alignment!")

    # Output bytes safely (FPDF returns str or bytearray depending on version)
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode('latin-1')
    return bytes(pdf_bytes)


# -----------------------------
# SUGGESTION ENGINE
# -----------------------------
def suggestion_for_skill(required_skill: str, closest_skill: str, similarity_pct: float) -> str:
    if similarity_pct >= 70:
        hint = "consolidate"
    elif similarity_pct >= 40:
        hint = "strengthen"
    else:
        hint = "learn"

    action = {
        "learn": "Add a focused project and mention tools/frameworks used.",
        "strengthen": "Improve depth with a mini-project and quantify outcomes.",
        "consolidate": "Show evidence (metrics, links) to solidify expertise."
    }[hint]

    if closest_skill:
        return f"{required_skill}: You have related experience in '{closest_skill}' (~{similarity_pct}%). {action}"
    return f"{required_skill}: Acquire baseline proficiency and showcase with a small project. {action}"


# -----------------------------
# SESSION & STATE
# -----------------------------
def init_session():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "username": None, "mode": "login"}
    if "results" not in st.session_state:
        st.session_state.results = None


# -----------------------------
# UI: AUTH PAGES
# -----------------------------
def ui_login():
    st.title("🔐 Login to Resume Analyzer")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        if verify_user(username, password):
            st.session_state.auth.update({"logged_in": True, "username": username})
            st.success("Welcome back! ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    st.info("New here?")
    if st.button("Create an account"):
        st.session_state.auth["mode"] = "signup"
        st.rerun()

def ui_signup():
    st.title("📝 Create an Account")
    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
    if submit:
        if not username.strip() or not password.strip():
            st.warning("Username and password cannot be empty.")
        elif password != confirm:
            st.warning("Passwords do not match.")
        elif create_user(username.strip(), password):
            st.success("Account created! Please log in.")
            st.session_state.auth["mode"] = "login"
            st.rerun()
        else:
            st.error("Username already exists. Try another.")

    if st.button("Back to Login"):
        st.session_state.auth["mode"] = "login"
        st.rerun()


# -----------------------------
# MAIN APP (ANALYZER)
# -----------------------------
def analyzer_app():
    nlp = load_spacy()
    model = load_st_model()

    # Sidebar
    with st.sidebar:
        st.success(f"👤 Logged in as **{st.session_state.auth['username']}**")
        if st.button("🚪 Logout"):
            st.session_state.auth = {"logged_in": False, "username": None, "mode": "login"}
            st.experimental_rerun()

        st.markdown("---")
        st.caption("Upload resume + job description to begin.")

    st.title("🧠 World-Class Resume Analyzer")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    with col2:
        jd_text = st.text_area("Paste Job Description", height=240, placeholder="Paste the full JD here…")

    if uploaded and jd_text:
        resume_text = read_resume_file(uploaded)
        if not resume_text.strip():
            st.error("Could not read resume text. Please check the file.")
            return

        # Extract entities
        candidate = extract_candidate_name(resume_text, nlp)
        job_title = extract_job_title(jd_text, nlp)

        # Skills
        resume_skills = get_skill_candidates(resume_text, nlp)
        jd_skills = get_skill_candidates(jd_text, nlp)

        # Semantic matching
        exact, mapping, score = semantic_skill_match(resume_skills, jd_skills, model)

        # Build dataframes
        mapping_df = pd.DataFrame(mapping)
        exact_matches = sorted(list(exact))
        missing_df = rank_missing(mapping)

        # Visuals
        matched_count = len(exact_matches)
        missing_count = max(0, len(dedup_preserve_order(jd_skills)) - matched_count)

        gcol, pcol = st.columns([1, 1])
        with gcol:
            gauge_fig = plot_gauge(score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        with pcol:
            donut_fig = plot_donut(matched_count, missing_count)
            st.plotly_chart(donut_fig, use_container_width=True)

        st.subheader("✅ Exact Matches")
        st.write(", ".join(exact_matches) if exact_matches else "No exact matches detected.")

        st.subheader("🔎 Required Skills Status (Semantic)")
        show_df = pd.DataFrame([
            {
                "Required Skill": m["jd_skill"],
                "Exact": "Yes" if m["exact"] else "No",
                "Closest Resume Skill": m["closest_resume_skill"],
                "Similarity (%)": round(m["similarity"]*100, 1)
            }
            for m in mapping
        ])
        st.dataframe(show_df, use_container_width=True)

        st.subheader("🚀 Enhancement Roadmap (Ranked)")
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            st.caption("Priority is based on semantic similarity: lower similarity → higher priority.")
        else:
            st.success("No missing skills. Great alignment!")

        # Suggestions (nice list)
        st.markdown("### 🛠️ Actionable Suggestions")
        if not missing_df.empty:
            for _, r in missing_df.head(10).iterrows():
                st.markdown(f"- {suggestion_for_skill(r['Required Skill'], r['Closest Resume Skill'], r['Similarity'])}")
        else:
            st.write("You're all set! Polish formatting, quantify impact, and add links to projects/portfolio.")

        # Generate PDF button
        with st.spinner("Preparing PDF…"):
            donut_path = None
            gauge_path = None
            try:
                # Save figures as images for PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_donut:
                    donut_fig.write_image(tmp_donut.name, engine="kaleido")
                    donut_path = tmp_donut.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_gauge:
                    gauge_fig.write_image(tmp_gauge.name, engine="kaleido")
                    gauge_path = tmp_gauge.name

                pdf_bytes = create_pdf(
                    candidate_name=candidate,
                    job_title=job_title,
                    username=st.session_state.auth["username"],
                    score=score,
                    donut_path=donut_path,
                    gauge_path=gauge_path,
                    matching_list=exact_matches,
                    missing_df=missing_df
                )
            finally:
                # Clean temp
                try:
                    if donut_path and os.path.exists(donut_path):
                        os.remove(donut_path)
                except Exception:
                    pass
                try:
                    if gauge_path and os.path.exists(gauge_path):
                        os.remove(gauge_path)
                except Exception:
                    pass

        st.download_button(
            label="📄 Download Professional Report (PDF)",
            data=pdf_bytes,
            file_name=f"{candidate.replace(' ','_')}_analysis.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Upload a resume and paste a job description to start the analysis.")


# -----------------------------
# BOOTSTRAP
# -----------------------------
def main():
    init_db()
    init_session()

    if not st.session_state.auth["logged_in"]:
        if st.session_state.auth["mode"] == "login":
            ui_login()
        else:
            ui_signup()
    else:
        analyzer_app()

if __name__ == "__main__":
    main()



