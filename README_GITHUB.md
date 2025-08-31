# 🚀 AI-Powered Resume Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-driven platform for automated resume analysis with semantic matching, sentiment analysis, global job recommendations, and professional PDF report generation.

## ✨ Features

### 🤖 AI-Powered Analysis
- **Semantic Similarity Matching** using advanced NLP
- **Skills Extraction & Categorization** 
- **Sentiment Analysis** of resume content
- **AI-Generated Insights** and recommendations

### 📊 Professional Reports
- **Interactive Visualizations** with Plotly charts
- **Downloadable PDF Reports** with comprehensive analysis
- **Skills Gap Analysis** with development suggestions
- **Match Score Calculations** with detailed breakdowns

### 🌍 Global Job Recommendations
- **Multi-Country Job Search** (US, UK, Canada, Australia, Germany)
- **Real-time Job Matching** based on skills
- **Remote Work Opportunities** 
- **Salary Information** and application links

### 📄 Multi-Format Support
- **PDF, DOCX, and TXT** file uploads
- **Advanced Text Extraction**
- **Error-Resistant Processing**

## 🚀 Quick Start

### Option 1: Try the Live Demo
👉 **[Launch Live App](https://your-app-url.streamlit.app)** *(Click here to try it now!)*

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app_working.py
```

4. **Open in browser**
```
http://localhost:8501
```

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
All required packages are listed in `requirements.txt`:
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations  
- **Pandas & NumPy** - Data processing
- **PyPDF2 & python-docx** - Document processing
- **TextBlob** - Natural language processing
- **Requests** - API integrations

## 🎯 How to Use

### 1. Upload Resume
- Drag and drop your resume (PDF, DOCX, or TXT)
- Enter candidate name and job title

### 2. Add Job Description  
- Paste the job description you want to match against
- Include specific skills and requirements

### 3. Analyze
- Click "🔍 ANALYZE RESUME" 
- View comprehensive AI analysis results
- Get match scores and insights

### 4. Generate Reports
- Click "📄 GENERATE & DOWNLOAD PDF REPORT"
- Download professional analysis report
- Share with hiring teams

### 5. Find Jobs
- Select target country and location
- Click "🔍 FIND JOBS" 
- Browse matching opportunities
- Apply directly through job links

## 📈 Technology Stack

### Frontend
- **Streamlit** - Interactive web interface
- **HTML/CSS** - Custom styling and animations
- **Plotly** - Data visualizations

### Backend  
- **Python** - Core application logic
- **TextBlob** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### APIs & Integrations
- **Adzuna Jobs API** - Real job listings
- **Remotive API** - Remote job opportunities  
- **OpenAI API** - Advanced AI analysis *(optional)*

## 🔧 Configuration

### API Setup (Optional)
To enable advanced features, add API keys:

1. **Create `.env` file:**
```env
OPENAI_API_KEY=your_openai_key
ADZUNA_APP_ID=your_adzuna_app_id  
ADZUNA_API_KEY=your_adzuna_key
```

2. **Get API Keys:**
- [OpenAI API](https://platform.openai.com/) - For advanced AI analysis
- [Adzuna API](https://developer.adzuna.com/) - For real job listings

## 🌟 Screenshots

### Main Interface
![Main Interface](screenshots/main-interface.png)

### Analysis Results  
![Analysis Results](screenshots/analysis-results.png)

### Job Recommendations
![Job Recommendations](screenshots/job-recommendations.png)

## 🚀 Deployment

### Deploy on Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from `app_working.py`

### Deploy on Heroku
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Deploy
git push heroku main
```

### Deploy on Railway
1. Connect GitHub repository to [Railway](https://railway.app)
2. Select this repository
3. Set main file as `app_working.py`
4. Deploy automatically

## 🎭 Demo Data

The application includes sample data for testing:
- **Sample resumes** for different job roles
- **Job descriptions** for various industries  
- **Multi-country job listings** with realistic salaries
- **Skills databases** covering 100+ technologies

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/ai-resume-analyzer.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Start development server
streamlit run app_working.py
```

## 🐛 Troubleshooting

### Common Issues

**Q: PDF generation not working?**
A: Install reportlab: `pip install reportlab`

**Q: Job search showing samples only?**  
A: Add Adzuna API credentials for real job listings

**Q: File upload failing?**
A: Check file size (max 200MB) and format (PDF/DOCX/TXT)

**Q: Slow performance?**
A: Reduce file size or upgrade to advanced AI mode

### Error Messages
- **"Could not extract text"** → Try different file format
- **"No matching skills"** → Update skills database  
- **"API rate limit"** → Wait or upgrade API plan

## 📞 Support

- **📧 Email:** support@nexgenai.com
- **💬 Issues:** [GitHub Issues](https://github.com/yourusername/ai-resume-analyzer/issues)
- **📚 Documentation:** [Wiki](https://github.com/yourusername/ai-resume-analyzer/wiki)
- **💡 Feature Requests:** [Discussions](https://github.com/yourusername/ai-resume-analyzer/discussions)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - Amazing web framework
- **OpenAI** - Advanced AI capabilities  
- **Adzuna** - Job listing API
- **Plotly** - Beautiful visualizations
- **Python Community** - Incredible ecosystem

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-resume-analyzer?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-resume-analyzer?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/ai-resume-analyzer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/ai-resume-analyzer)

---

⭐ **If this project helped you, please give it a star!** ⭐

**Made with ❤️ for better recruitment processes**
