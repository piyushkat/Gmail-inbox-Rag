import torch
# ✅ Manually fix the torch.classes issue
torch.classes.__path__ = []  
import os
import pandas as pd
import streamlit as st
import chromadb
from auth.gmail_auth import authenticate_gmail
from data.email_fetcher import get_emails
from data.vectordb import create_vector_database
from analytics.email_eda import perform_eda
from rag.qa_engine import ask_question_with_groq
from utils.text_processing import clean_text

def streamlit_app():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Gmail RAG System",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'service' not in st.session_state:
        st.session_state.service = None
    if 'emails_df' not in st.session_state:
        st.session_state.emails_df = None
    if 'vector_db' not in st.session_state:
        try:
            os.makedirs("./chroma_db", exist_ok=True)
            st.session_state.vector_db = chromadb.PersistentClient(path="./chroma_db")
            st.session_state.vector_db_initialized = True
            st.info("ChromaDB client initialized with persistent storage.")
        except Exception as e:
            st.error(f"Failed to initialize Chroma client: {str(e)}")
            st.session_state.vector_db = None
            st.session_state.vector_db_initialized = False
    if 'eda_done' not in st.session_state:
        st.session_state.eda_done = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "qa"  # Default view is the Q&A interface
    
    # Sidebar for authentication and data loading
    with st.sidebar:
        st.title("Gmail RAG System")
        st.subheader("with Groq Integration")
        
        # Authentication section
        st.header("1. Gmail Authentication")
        if not st.session_state.authenticated:
            if st.button("Authenticate with Gmail"):
                with st.spinner("Authenticating..."):
                    service = authenticate_gmail()
                    if service:
                        st.session_state.service = service
                        st.session_state.authenticated = True
                        st.success("Authentication successful!")
                    else:
                        st.error("Authentication failed. Please try again.")
        else:
            st.success("✅ Authenticated with Gmail")
        
        # Data loading section
        st.header("2. Load Emails")
        if st.session_state.authenticated and st.session_state.emails_df is None:
            if st.button("Fetch All Emails"):
                progress_bar = st.progress(0)
                emails = get_emails(st.session_state.service, progress_bar=progress_bar)
                if emails:
                    st.session_state.emails_df = pd.DataFrame(emails)
                    st.session_state.emails_df['clean_body'] = st.session_state.emails_df['body'].apply(clean_text)
                    st.success(f"✅ Loaded {len(emails)} emails")
                else:
                    st.error("Failed to fetch emails")
        elif st.session_state.emails_df is not None:
            st.success(f"✅ Loaded {len(st.session_state.emails_df)} emails")
        
        # Vector DB creation
        st.header("3. Create Vector Database")
        if st.session_state.emails_df is not None:
            if st.session_state.vector_db is None:
                st.error("Chroma client not initialized. Please reset the app.")
            elif not hasattr(st.session_state.vector_db, 'collection_created'):
                if st.button("Create Vector Database"):
                    progress_bar = st.progress(0)
                    success = create_vector_database(st.session_state.emails_df, st.session_state.vector_db, progress_bar=progress_bar)
                    if success:
                        st.session_state.vector_db.collection_created = True
                        st.success("✅ Vector database created")
                    else:
                        st.error("Failed to create vector database")
            else:
                st.success("✅ Vector database ready")
        else:
            st.info("Load emails first to create the vector database.")
        
        # Navigation section
        st.header("4. Navigation")
        if st.session_state.emails_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Q&A Interface"):
                    st.session_state.current_view = "qa"
                    st.rerun()
            with col2:
                if st.button("Analyze Emails"):
                    st.session_state.current_view = "eda"
                    st.session_state.eda_done = True
                    st.rerun()
        
        # Reset application
        st.header("5. Reset Application")
        if st.button("Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    if st.session_state.current_view == "eda" and st.session_state.emails_df is not None:
        # Display EDA results with back button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Back to Q&A"):
                st.session_state.current_view = "qa"
                st.rerun()
        
        # Perform email analysis
        perform_eda(st.session_state.emails_df)
    
    # Q&A interface
    elif st.session_state.current_view == "qa":
        if st.session_state.vector_db is not None and hasattr(st.session_state.vector_db, 'collection_created'):
            # Use columns to center the title and input box
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.title("Ask Questions About Your Emails")
                
                # Input box for questions with larger width
                question = st.text_input("Ask a question about your emails:", key="question_input")
                
                # Submit button
                if st.button("Ask Question") and question:
                    # Process the question and get an answer
                    result = ask_question_with_groq(question, st.session_state.vector_db)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    
                    # Display sources
                    st.subheader("Sources:")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"Source {i+1}: {source['metadata']['subject']}"):
                            st.write(f"From: {source['metadata']['from']}")
                            st.write(f"Date: {source['metadata']['date']}")
                            st.write("Excerpt:")
                            st.write(source['excerpt'])
                    
                    # Store in chat history
                    st.session_state.chat_history.append({"question": question, "result": result})
                
                # Display chat history
                if st.session_state.chat_history:
                    st.subheader("Previous Questions")
                    for i, chat in enumerate(st.session_state.chat_history):
                        with st.expander(f"Q: {chat['question']}"):
                            st.write("A: " + chat['result']['answer'])
        else:
            # Welcome screen centered in the main area
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.title("Gmail RAG System Setup")
                st.info("Please complete the steps in the sidebar to set up your email analysis system.")

if __name__ == "__main__":
    streamlit_app()