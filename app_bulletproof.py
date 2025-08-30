# =============================================================================
# 🚀 NEXGEN AI RESUME ANALYZER - BULLETPROOF VERSION
# World-Class Resume Analysis Platform - 403 Error Fixed
# =============================================================================

import streamlit as st
import os
import sqlite3
import hashlib
import datetime
import json
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Document Processing
import PyPDF2
import docx

# Advanced AI/NLP
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from textblob import TextBlob
    import openai
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False

# PDF Report Generation