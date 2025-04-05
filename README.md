# Gmail Inbox RAG System with Groq Integration

A powerful Gmail inbox analysis tool that combines RAG (Retrieval-Augmented Generation) with Groq's LLM to provide intelligent email analysis and question-answering capabilities.

## 🌟 Features

- **Gmail Integration**: Secure OAuth2.0 authentication with Gmail API
- **Email Analysis**: 
  - Fetch and process emails from your Gmail account
  - Clean and normalize email content
  - Extract metadata (sender, date, subject)
- **Vector Database**: 
  - ChromaDB integration for efficient email storage
  - Sentence transformer embeddings
  - Persistent storage for future sessions
- **Interactive Q&A**: 
  - Natural language questions about your emails
  - Groq LLM integration for accurate answers
  - Source attribution for answers
- **Email Analytics (EDA)**:
  - Email volume trends
  - Top sender analysis
  - Word frequency analysis
  - Email length distribution
  - Topic modeling
- **User Interface**: 
  - Streamlit-based web interface
  - Progress tracking for long operations
  - Chat history management

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Vector Database**: ChromaDB
- **LLM Integration**: Groq API
- **Email Service**: Gmail API
- **Key Libraries**:
  - `torch`: For machine learning operations
  - `sentence-transformers`: For text embeddings
  - `pandas`: For data manipulation
  - `nltk`: For text processing
  - `scikit-learn`: For topic modeling
  - `wordcloud`: For text visualization
  - `beautifulsoup4`: For HTML parsing
  - `matplotlib`: For data visualization

## 📋 Prerequisites

1. Python 3.8 or higher
2. Google Cloud Platform account with Gmail API enabled
3. Groq API key
4. Chrome browser (recommended for OAuth flow)

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gmail-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud Platform**
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download the client configuration file as `client_secret.json`

4. **Configure Groq API**
   - Get your Groq API key
   - Set it in `rag/qa_engine.py`

5. **Run the application**
   ```bash
   streamlit run gmail_rag_system/app.py
   ```

## 📁 Project Structure

```
gmail_rag_system/
├── app.py                 # Main Streamlit application
├── auth/
│   └── gmail_auth.py     # Gmail authentication module
├── data/
│   ├── email_fetcher.py  # Email fetching functionality
│   └── vectordb.py       # Vector database operations
├── analytics/
│   └── email_eda.py      # Email analysis and visualization
├── rag/
│   └── qa_engine.py      # RAG implementation with Groq
├── utils/
│   └── text_processing.py # Text cleaning utilities
└── chroma_db/            # Persistent vector database storage
```

## 🔒 Security Features

- OAuth 2.0 authentication for Gmail
- Secure token storage
- No permanent email content storage
- Local vector database storage
- Environment variable support for API keys

## 💡 Usage Guide

1. **Authentication**
   - Click "Authenticate with Gmail" in the sidebar
   - Complete the OAuth flow in your browser
   - Grant necessary permissions

2. **Email Loading**
   - Use "Fetch All Emails" to load your inbox
   - Watch the progress bar for status
   - View success confirmation

3. **Vector Database Creation**
   - Click "Create Vector Database" after emails are loaded
   - Wait for the embedding process to complete
   - Confirm database creation success

4. **Ask Questions**
   - Use the Q&A interface to ask about your emails
   - View answers with source attribution
   - Access chat history for previous questions

5. **Email Analysis**
   - Switch to "Analyze Emails" view
   - Explore various visualizations
   - View topic modeling results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.  