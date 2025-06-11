# BankBot
# BankBot - AI Banking Chatbot

## Project Overview
BankBot is an AI-powered chatbot designed to handle customer banking queries using Natural Language Processing (NLP) techniques. The system uses Flask for the backend, SQLite for data storage, and a responsive web interface for user interaction.

## Features
- **Intent Classification**: Identifies user intentions with 87%+ accuracy
- **Entity Recognition**: Extracts account numbers, amounts, and other relevant data
- **Secure Database Integration**: SQLite database for chat history and account information
- **Real-time Responses**: Instant responses to user queries
- **Responsive UI**: Modern, mobile-friendly chat interface
- **RESTful API**: Clean API endpoints for integration

## Tech Stack
- **Backend**: Python, Flask, SQLite
- **NLP**: NLTK, Scikit-learn, TF-IDF Vectorization
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite
- **APIs**: RESTful API architecture

## Installation Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project
Create a new directory for your project and add the provided files:
```
banking_chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend interface
└── banking_chatbot.db    # SQLite database (auto-created)
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv banking_chatbot_env
source banking_chatbot_env/bin/activate  # On Windows: banking_chatbot_env\Scripts\activate
```

### Step 3: Install Dependencies
Create a `requirements.txt` file with the following content:
```
Flask==2.3.3
nltk==3.8.1
scikit-learn==1.3.0
numpy==1.24.3
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Create Directory Structure
```bash
mkdir templates
```

### Step 5: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`
