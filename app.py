from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import datetime
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

class BankingChatbot:
    def __init__(self):
        self.intents = self.load_intents()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
        self.init_database()
        
    def load_intents(self):
        """Load predefined banking intents and responses"""
        return {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "responses": ["Hello! Welcome to our banking service. How can I help you today?",
                            "Hi there! I'm here to assist you with your banking needs.",
                            "Good day! How may I assist you with your banking queries?"]
            },
            "balance_inquiry": {
                "patterns": ["what is my balance", "check balance", "account balance", "how much money do I have",
                           "balance check", "show my balance", "current balance"],
                "responses": ["I can help you check your account balance. Please provide your account number.",
                            "To check your balance, I'll need to verify your account details first.",
                            "Let me help you with your balance inquiry. What's your account number?"]
            },
            "transfer_money": {
                "patterns": ["transfer money", "send money", "wire transfer", "money transfer", "transfer funds",
                           "move money", "transfer to account"],
                "responses": ["I can help you transfer money. Please provide the recipient's account details.",
                            "For money transfers, I'll need the destination account number and amount.",
                            "Let me assist you with the money transfer. What amount would you like to send?"]
            },
            "loan_inquiry": {
                "patterns": ["loan application", "apply for loan", "personal loan", "home loan", "car loan",
                           "loan eligibility", "loan rates", "loan information"],
                "responses": ["I can provide information about our loan products. What type of loan are you interested in?",
                            "We offer various loan options. Would you like to know about personal, home, or auto loans?",
                            "Let me help you with loan information. What's your preferred loan type?"]
            },
            "card_services": {
                "patterns": ["credit card", "debit card", "card blocked", "card activation", "new card",
                           "card replacement", "card limit", "card statement"],
                "responses": ["I can help you with card-related services. What do you need assistance with?",
                            "For card services, I can help with activation, replacement, or general inquiries.",
                            "Let me assist you with your card needs. Please describe your issue."]
            },
            "branch_location": {
                "patterns": ["branch location", "nearest branch", "atm location", "bank address",
                           "find branch", "branch timings", "office hours"],
                "responses": ["I can help you find our nearest branch or ATM. What's your current location?",
                            "Let me help you locate our branches. Which area are you looking for?",
                            "I can provide branch locations and timings. What city are you in?"]
            },
            "complaint": {
                "patterns": ["complaint", "issue", "problem", "error", "wrong transaction", "dispute",
                           "not working", "help me", "support"],
                "responses": ["I'm sorry to hear about the issue. Let me help you resolve this complaint.",
                            "I understand your concern. Please provide more details about the problem.",
                            "Let me assist you with this issue. Can you describe what happened?"]
            },
            "goodbye": {
                "patterns": ["bye", "goodbye", "see you later", "thanks", "thank you", "that's all"],
                "responses": ["Thank you for using our banking service. Have a great day!",
                            "Goodbye! Feel free to contact us anytime for banking assistance.",
                            "Thank you for your time. We're always here to help with your banking needs."]
            }
        }
    
    def init_database(self):
        """Initialize SQLite database for storing chat history and user data"""
        conn = sqlite3.connect('banking_chatbot.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                bot_response TEXT,
                intent TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_accounts (
                account_number TEXT PRIMARY KEY,
                account_holder TEXT,
                balance REAL,
                account_type TEXT
            )
        ''')
        
        # Insert sample account data
        sample_accounts = [
            ('123456789', 'John Doe', 5000.00, 'Savings'),
            ('987654321', 'Jane Smith', 15000.00, 'Checking'),
            ('555666777', 'Mike Johnson', 8500.50, 'Savings')
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO user_accounts 
            (account_number, account_holder, balance, account_type) 
            VALUES (?, ?, ?, ?)
        ''', sample_accounts)
        
        conn.commit()
        conn.close()
    
    def preprocess_text(self, text):
        """Preprocess user input text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters except spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize and lemmatize
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    def train_model(self):
        """Train the intent classification model"""
        training_data = []
        training_labels = []
        
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                processed_pattern = self.preprocess_text(pattern)
                training_data.append(processed_pattern)
                training_labels.append(intent)
        
        # Vectorize the training data
        X = self.vectorizer.fit_transform(training_data)
        
        # Train the classifier
        self.classifier.fit(X, training_labels)
        self.is_trained = True
        
        print("Model trained successfully!")
    
    def predict_intent(self, user_input):
        """Predict the intent of user input"""
        if not self.is_trained:
            self.train_model()
        
        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        
        # Get prediction and confidence
        intent = self.classifier.predict(input_vector)[0]
        confidence = max(self.classifier.predict_proba(input_vector)[0])
        
        return intent, confidence
    
    def extract_entities(self, user_input):
        """Extract entities like account numbers, amounts, etc."""
        entities = {}
        
        # Extract account numbers (9-12 digits)
        account_pattern = r'\b\d{9,12}\b'
        accounts = re.findall(account_pattern, user_input)
        if accounts:
            entities['account_number'] = accounts[0]
        
        # Extract monetary amounts
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(amount_pattern, user_input)
        if amounts:
            entities['amount'] = amounts[0].replace(',', '')
        
        return entities
    
    def get_account_balance(self, account_number):
        """Retrieve account balance from database"""
        conn = sqlite3.connect('banking_chatbot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT account_holder, balance, account_type 
            FROM user_accounts 
            WHERE account_number = ?
        ''', (account_number,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'account_holder': result[0],
                'balance': result[1],
                'account_type': result[2]
            }
        return None
    
    def save_chat_history(self, user_message, bot_response, intent):
        """Save chat interaction to database"""
        conn = sqlite3.connect('banking_chatbot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (user_message, bot_response, intent)
            VALUES (?, ?, ?)
        ''', (user_message, bot_response, intent))
        
        conn.commit()
        conn.close()
    
    def generate_response(self, user_input):
        """Generate appropriate response based on user input"""
        # Predict intent
        intent, confidence = self.predict_intent(user_input)
        
        # Extract entities
        entities = self.extract_entities(user_input)
        
        # Handle low confidence predictions
        if confidence < 0.6:
            response = "I'm not sure I understand. Could you please rephrase your question or provide more details?"
        else:
            # Handle specific intents
            if intent == "balance_inquiry" and "account_number" in entities:
                account_info = self.get_account_balance(entities["account_number"])
                if account_info:
                    response = f"Hello {account_info['account_holder']}, your {account_info['account_type'].lower()} account balance is ${account_info['balance']:,.2f}."
                else:
                    response = "I couldn't find an account with that number. Please verify your account number."
            else:
                # Get random response from intent responses
                responses = self.intents[intent]["responses"]
                response = random.choice(responses)
        
        # Save to chat history
        self.save_chat_history(user_input, response, intent)
        
        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        }

# Initialize chatbot
chatbot = BankingChatbot()

@app.route('/')
def index():
    """Render main chatbot interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        result = chatbot.generate_response(user_message)
        
        return jsonify({
            'response': result['response'],
            'intent': result['intent'],
            'confidence': float(result['confidence']),
            'entities': result['entities']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account/<account_number>')
def get_account(account_number):
    """API endpoint to get account information"""
    try:
        account_info = chatbot.get_account_balance(account_number)
        if account_info:
            return jsonify(account_info)
        else:
            return jsonify({'error': 'Account not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat_history')
def get_chat_history():
    """API endpoint to get chat history"""
    try:
        conn = sqlite3.connect('banking_chatbot.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, bot_response, intent, timestamp 
            FROM chat_history 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'user_message': row[0],
                'bot_response': row[1],
                'intent': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)