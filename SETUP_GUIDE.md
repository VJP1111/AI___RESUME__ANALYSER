# =============================================================================
# 🚀 NEXGEN AI RESUME ANALYZER - SETUP GUIDE
# =============================================================================

## API Keys Setup (Optional but Recommended)

To enable full AI and job recommendation features, set up these API keys:

### 1. OpenAI API (for LLM Analysis)
- Sign up at: https://platform.openai.com/
- Get your API key
- Set environment variable: OPENAI_API_KEY=your_key_here

### 2. Adzuna Job API (for Job Recommendations)
- Sign up at: https://developer.adzuna.com/
- Get App ID and API Key
- Set environment variables:
  - ADZUNA_APP_ID=your_app_id
  - ADZUNA_API_KEY=your_api_key

### 3. Setting Environment Variables (Windows)

#### Method 1: PowerShell (Temporary)
```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:ADZUNA_APP_ID="your_adzuna_app_id"
$env:ADZUNA_API_KEY="your_adzuna_api_key"
```

#### Method 2: Create .env file
Create a file named `.env` in your project folder:
```
OPENAI_API_KEY=your_openai_key_here
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_API_KEY=your_adzuna_api_key
```

#### Method 3: System Environment Variables
1. Open System Properties > Environment Variables
2. Add the variables there

### 4. Installation Commands

```bash
# Basic installation
pip install -r requirements.txt

# For full AI features (optional)
pip install sentence-transformers torch

# For advanced LLM features
pip install openai

# For PDF generation
pip install reportlab
```

### 5. Running the Application

```bash
# Navigate to project directory
cd "c:\Users\komal\OneDrive\Desktop\PROJECT RESUME ANALYSER"

# Activate virtual environment (if using)
venv\Scripts\activate

# Run the enhanced application
streamlit run app_bulletproof_enhanced.py
```

### 6. Features Available

#### Without API Keys:
- ✅ Resume analysis with basic AI
- ✅ Skills extraction and matching
- ✅ PDF report generation
- ✅ Sample job recommendations
- ✅ Basic insights

#### With API Keys:
- 🚀 Advanced LLM analysis (OpenAI GPT)
- 🚀 Real-time job recommendations (Adzuna)
- 🚀 Enhanced AI insights
- 🚀 Professional career recommendations
- 🚀 Semantic similarity matching

### 7. Troubleshooting

If you encounter issues:
1. Make sure all dependencies are installed
2. Check API key validity
3. Verify internet connection for API calls
4. Check the terminal for error messages

For support, check the application logs or contact support.