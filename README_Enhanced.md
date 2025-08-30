# Resume Analyzer Pro 🎯

An advanced AI-powered resume analysis tool with comprehensive insights and professional reporting capabilities.

## Features ✨

### 🔐 Authentication System
- **Sidebar Login/Signup**: Easy authentication in the sidebar
- **User Profiles**: Personal dashboard and profile management
- **Secure Authentication**: Password hashing and user sessions

### 🤖 Advanced AI Analysis
- **Semantic Matching**: Uses NLP models for intelligent resume-job matching
- **Skills Extraction**: Automatically identifies technical skills from resume and job description
- **Smart Scoring**: Calculates match percentage based on skill alignment
- **Enhancement Suggestions**: Actionable recommendations for improvement

### 📊 Visual Analytics
- **Interactive Gauge Charts**: Visual match score representation
- **Pie Charts**: Skills distribution analysis
- **Category Breakdown**: Skills organized by technology categories
- **Responsive Design**: Works on all devices

### 📄 Professional Reports
- **PDF Generation**: Comprehensive analysis reports
- **Data Export**: JSON export for further analysis
- **Professional Formatting**: Clean, structured reports

### 💾 Data Management
- **Analysis History**: Save and track all analyses
- **User Dashboard**: View past analyses and statistics
- **Profile Management**: Update personal information
- **SQLite Database**: Local data persistence

## Installation & Setup 🚀

1. **Clone or Download** the project files
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app_enhanced.py
   ```

4. **Access the App**: Open your browser to `http://localhost:8501`

## File Structure 📁

```
PROJECT RESUME ANALYSER/
├── app_enhanced.py          # Main enhanced application
├── app.py                   # Original application (backup)
├── app_upgraded.py          # Alternative version (backup)
├── requirements.txt         # Python dependencies
├── auth.py                  # Authentication utilities (legacy)
├── core.py                  # Database models (legacy)
├── users.json              # User data (legacy)
├── uploads/                # Uploaded resume files
├── reports/                # Generated PDF reports
└── resume_analyzer.db      # SQLite database
```

## How to Use 📖

### 1. Create Account / Login
- Use the sidebar to create a new account or login
- Fill in your username, email, and password
- Login to access all features

### 2. Analyze Resume
- Navigate to the "🎯 Analyzer" tab
- Upload your resume (PDF, DOCX, or TXT)
- Paste the job description
- Click "🔍 Analyze Resume"

### 3. View Results
- **Match Score**: See your overall compatibility percentage
- **Skills Breakdown**: View matching and missing skills
- **Visual Charts**: Interactive gauge and pie charts
- **Suggestions**: Get actionable improvement recommendations

### 4. Generate Reports
- Click "📄 Generate PDF Report" for professional documentation
- Download and save reports for your records
- Export data as JSON for further analysis

### 5. Track Progress
- View your analysis history in the "📊 Dashboard" tab
- See statistics and trends over time
- Access previous analyses anytime

### 6. Manage Profile
- Update your profile information in the "👤 Profile" tab
- View account statistics
- Manage your personal data

## Technology Stack 🛠️

- **Frontend**: Streamlit (Web Interface)
- **Backend**: Python, SQLite Database
- **AI/NLP**: Sentence Transformers, Scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Document Processing**: PyPDF2, python-docx
- **Report Generation**: FPDF2

## Features Implemented ✅

- ✅ **Sidebar Login/Signup/Logout**
- ✅ **Advanced AI & NLP Analysis**
- ✅ **Interactive Pie Charts & Gauge Visualizations**
- ✅ **Comprehensive PDF Report Generation**
- ✅ **User Profile & Dashboard**
- ✅ **Missing Skills Analysis with Enhancement Suggestions**
- ✅ **Data Persistence & Analysis History**
- ✅ **Responsive Design**
- ✅ **Professional UI/UX**

## Support 🆘

If you encounter any issues:

1. **Check Dependencies**: Ensure all packages in `requirements.txt` are installed
2. **Python Version**: Use Python 3.8 or higher
3. **File Permissions**: Ensure the app can create folders and database files
4. **Browser Compatibility**: Use a modern web browser

## Future Enhancements 🔮

Potential future improvements:
- Integration with job boards APIs
- Multi-language resume support
- Advanced ML models for better matching
- Resume template generation
- LinkedIn integration
- Team collaboration features

## License 📝

This project is for educational and personal use. Feel free to modify and enhance according to your needs.

---

**Built with ❤️ for better career outcomes!**

*Version: Enhanced Pro - January 2025*