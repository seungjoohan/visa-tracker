# 🛂 Visa News Tracker

AI-powered visa policy news tracking and analysis system that collects, analyzes, and delivers daily email summaries of visa-related news.

## ✨ Features

- **Automated News Collection**: Gathers visa-related news from NewsAPI
- **Intelligent Classification**: Categorizes news by importance level
  - 🚨 **Needs Attention**: Urgent policy changes requiring immediate attention
  - 📌 **Good to Know**: Important updates worth knowing about
  - 📰 **No Attention Required**: General background information
- **Smart Filtering**: Filters news by relevance to visa/immigration topics
- **Daily Email Summaries**: Beautiful HTML email reports with organized news
- **Modular FastAPI Architecture**: Scalable design ready for future enhancements

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd visa-tracker

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp env.example .env
```

```env
# NewsAPI Configuration
NEWSAPI_KEY=your_newsapi_key_here

# Email Configuration
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_RECIPIENTS=["recipient1@example.com", "recipient2@example.com"]

# SMTP Settings (Gmail example)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 3. Get API Keys

- **NewsAPI**: Sign up at [newsapi.org](https://newsapi.org) for free API key (1000 requests/day)
- **Gmail App Password**: Enable 2FA and generate an app password for SMTP

### 4. Run Daily Processing

```bash
# Manual run for testing
python scripts/run_daily.py

# Or start FastAPI server (optional)
uvicorn app.main:app --reload
```

### 5. Set Up Automated Schedule (Github Actions)

```bash
# 1. GitHub Repository 생성 및 업로드
git add .
git commit -m "feat: Complete visa news tracker"
git remote add origin https://github.com/yourusername/visa-tracker.git
git push -u origin main

# 2. GitHub Secrets 설정 (Repository Settings → Secrets)
# NEWSAPI_KEY, EMAIL_USER, EMAIL_PASSWORD, EMAIL_RECIPIENTS
```

## 📁 Project Structure

```

visa-tracker/
├── app/
│ ├── core/
│ │ ├── config.py # Configuration management
│ │ └── dependencies.py # Dependency injection
│ ├── models/
│ │ ├── news.py # Data models
│ │ └── email.py # Email models
│ ├── services/
│ │ ├── news_collector.py # News collection services
│ │ ├── news_analyzer.py # Analysis and classification
│ │ └── email_service.py # Email handling
│ ├── tasks/
│ │ └── daily_processor.py # Main processing pipeline
│ └── main.py # FastAPI application
├── config/
│ └── email_template.html # Email template
├── scripts/
│ └── run_daily.py # CLI execution script
└── requirements.txt # Dependencies

```

## 🔮 Future Enhancements

- **WhiteHouse.gov Integration**: Official policy announcements
- **Web Dashboard**: Interactive news browsing interface
- **User Management**: Multiple subscribers with personalized preferences
- **Advanced NLP**: More sophisticated text analysis
- **Real-time Alerts**: WebSocket notifications for urgent news
- **API Endpoints**: RESTful API for third-party integrations

## ⚙️ Configuration Options

You can customize behavior through environment variables:

```env
# News Collection
MAX_ARTICLES_PER_SOURCE=50
RELEVANCE_THRESHOLD=0.5

# Classification Keywords (JSON arrays)
URGENT_KEYWORDS=["ban", "suspended", "emergency"]
IMPORTANT_KEYWORDS=["policy", "change", "update"]
```
