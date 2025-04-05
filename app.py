import torch
# ✅ Manually fix the torch.classes issue
torch.classes.__path__ = []  
import os
import pandas as pd
import streamlit as st
import time
import threading
import schedule
import chromadb
from dotenv import load_dotenv

# Import functions from our modules
from auth.gmail_auth import authenticate_gmail
from data.email_fetcher import get_emails, fetch_new_emails
from data.vectordb import create_vector_database, update_vector_database
from analytics.email_eda import perform_eda
from rag.qa_engine import ask_question_with_groq
from utils.text_processing import clean_text

# Load environment variables
load_dotenv()

def scheduler_thread():
    """Background thread to run the scheduler."""
    while st.session_state.get('scheduler_running', False):
        schedule.run_pending()
        time.sleep(1)

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
    if 'last_email_date' not in st.session_state:
        st.session_state.last_email_date = None
    if 'scheduler_running' not in st.session_state:
        st.session_state.scheduler_running = False
    if 'scheduler_thread' not in st.session_state:
        st.session_state.scheduler_thread = None
    
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
                    
                    # Set the last email date for future fetches
                    if 'date' in st.session_state.emails_df.columns:
                        max_date = pd.to_datetime(st.session_state.emails_df['date'], errors='coerce').max()
                        if pd.notna(max_date):
                            st.session_state.last_email_date = max_date
                            st.info(f"Last email date recorded: {max_date}")
        elif st.session_state.emails_df is not None:
            st.success(f"✅ {len(st.session_state.emails_df)} emails loaded")
            
            # Button to fetch new emails
            if st.button("Fetch New Emails"):
                progress_bar = st.progress(0)
                new_emails = get_emails(st.session_state.service, 
                                      progress_bar=progress_bar, 
                                      after_date=st.session_state.last_email_date)
                if new_emails:
                    new_emails_df = pd.DataFrame(new_emails)
                    new_emails_df['clean_body'] = new_emails_df['body'].apply(clean_text)
                    
                    # Update the DataFrame
                    combined_df = pd.concat([st.session_state.emails_df, new_emails_df])
                    st.session_state.emails_df = combined_df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
                    
                    # Update the vector database with new emails
                    if st.session_state.get('vector_db_created', False):
                        success = update_vector_database(new_emails_df, st.session_state.vector_db)
                        if success:
                            st.success("Vector database updated with new emails")
                        else:
                            st.error("Failed to update vector database")
                    
                    # Update the last email date
                    if len(new_emails_df) > 0 and 'date' in new_emails_df.columns:
                        max_date = pd.to_datetime(new_emails_df['date'], errors='coerce').max()
                        if pd.notna(max_date):
                            st.session_state.last_email_date = max_date
                            st.info(f"Updated last email date to {max_date}")
                    
                    st.success(f"Added {len(new_emails)} new emails")
                else:
                    st.info("No new emails found")
            
            # Set up auto-refresh
            st.header("Auto-Refresh Settings")
            enable_auto_refresh = st.checkbox("Enable auto-refresh", value=st.session_state.get('scheduler_running', False))
            
            refresh_interval = st.selectbox(
                "Refresh interval",
                ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
                index=1  # Default to 5 minutes
            )
            
            # Handle scheduler based on checkbox
            if enable_auto_refresh != st.session_state.get('scheduler_running', False):
                if enable_auto_refresh:
                    # Start the scheduler
                    st.session_state.scheduler_running = True
                    
                    # Clear any existing schedules
                    schedule.clear()
                    
                    # Set up the schedule based on selected interval
                    if refresh_interval == "1 minute":
                        schedule.every(1).minutes.do(fetch_new_emails)
                    elif refresh_interval == "5 minutes":
                        schedule.every(5).minutes.do(fetch_new_emails)
                    elif refresh_interval == "15 minutes":
                        schedule.every(15).minutes.do(fetch_new_emails)
                    elif refresh_interval == "30 minutes":
                        schedule.every(30).minutes.do(fetch_new_emails)
                    else:  # 1 hour
                        schedule.every(1).hours.do(fetch_new_emails)
                    
                    # Start the scheduler thread
                    st.session_state.scheduler_thread = threading.Thread(target=scheduler_thread)
                    st.session_state.scheduler_thread.daemon = True
                    st.session_state.scheduler_thread.start()
                    
                    st.success(f"Auto-refresh enabled with {refresh_interval} interval")
                else:
                    # Stop the scheduler
                    st.session_state.scheduler_running = False
                    st.info("Auto-refresh disabled")
        
        # Vector database section
        st.header("3. Create Vector Database")
        if st.session_state.emails_df is not None and st.session_state.vector_db_initialized:
            if not st.session_state.get('vector_db_created', False):
                if st.button("Create Vector Database"):
                    progress_bar = st.progress(0)
                    success = create_vector_database(st.session_state.emails_df, st.session_state.vector_db, progress_bar)
                    if success:
                        st.session_state.vector_db_created = True
                        st.success("Vector database created successfully!")
                    else:
                        st.error("Failed to create vector database.")
            else:
                st.success("✅ Vector database created")
                
                # Option to recreate database
                if st.button("Recreate Vector Database"):
                    progress_bar = st.progress(0)
                    success = create_vector_database(st.session_state.emails_df, st.session_state.vector_db, progress_bar)
                    if success:
                        st.success("Vector database recreated successfully!")
                    else:
                        st.error("Failed to recreate vector database.")
    
    # Main content area
    if st.session_state.emails_df is not None:
        # Tabs for different features
        tab1, tab2 = st.tabs(["Email Analysis", "Email Search & Chat"])
        
        # Tab 1: Email Analysis
        with tab1:
            if not st.session_state.eda_done:
                eda_results = perform_eda(st.session_state.emails_df)
                if eda_results and 'sender_counts' in eda_results and eda_results['sender_counts'] is not None:
                    st.session_state.sender_counts = eda_results['sender_counts']
                st.session_state.eda_done = True
            else:
                if st.button("Refresh Analysis"):
                    st.session_state.eda_done = False
                    st.experimental_rerun()
                else:
                    eda_results = perform_eda(st.session_state.emails_df)
        
        # Tab 2: Email Search & Chat
        with tab2:
            st.header("Email Assistant", divider="rainbow")
            
            if not st.session_state.get('vector_db_created', False):
                st.warning("Please create the vector database before using this feature.")
            else:
                st.markdown("**Ask a question about your emails:**")
                user_question = st.text_input("Question:", key="rag_question")
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    ask_button = st.button("Ask")
                with col2:
                    clear_button = st.button("Clear Chat")
                
                if clear_button:
                    st.session_state.chat_history = []
                    st.rerun()
                
                if ask_button and user_question:
                    # Add user question to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    
                    # Get answer from RAG system
                    response = ask_question_with_groq(user_question, st.session_state.vector_db)
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**Assistant:** {message['content']}")
                        
                        # Show sources if available
                        if "sources" in message:
                            with st.expander("View Sources"):
                                for i, source in enumerate(message["sources"]):
                                    meta = source["metadata"]
                                    st.markdown(f"**Email {i+1}**")
                                    st.markdown(f"**From:** {meta['from']}")
                                    st.markdown(f"**Date:** {meta['date']}")
                                    st.markdown(f"**Subject:** {meta['subject']}")
                                    st.markdown(f"**Excerpt:** {source['excerpt']}")
                                    st.markdown("---")
                    
                    st.markdown("---")

if __name__ == "__main__":
    streamlit_app()