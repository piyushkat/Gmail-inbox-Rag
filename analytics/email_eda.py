import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import streamlit as st

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def perform_eda(emails_df):
    """Perform Exploratory Data Analysis on the email dataset."""
    st.header("Email Analysis Dashboard", divider="rainbow")
    
    # Create output directory for plots
    os.makedirs('eda_output', exist_ok=True)
    
    # Make a copy of the dataframe to avoid modifying the original
    eda_df = emails_df.copy()
    sender_counts = None  # Initialize for return value
    
    # Basic statistics
    st.subheader("Quick Stats")
    
    total_emails = len(eda_df)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Emails", f"{total_emails:,}")
    
    # Date range
    if 'date' in eda_df.columns and eda_df['date'].notna().any():
        try:
            eda_df['date'] = pd.to_datetime(eda_df['date'], errors='coerce')
            earliest = eda_df['date'].min()
            latest = eda_df['date'].max()
            date_range = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
            with col2:
                st.metric("Date Range", date_range)
        except:
            pass
    
    # 1. Email Volume Over Time
    st.subheader("Email Volume Over Time")
    if 'date' in eda_df.columns and eda_df['date'].notna().any():
        try:
            # Ensure date is datetime
            eda_df['date'] = pd.to_datetime(eda_df['date'], errors='coerce')
            
            # Handle timezone issues by standardizing to UTC
            eda_df['date_normalized'] = eda_df['date'].apply(
                lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo is not None else x
            )
            
            # Extract date component safely
            eda_df['date_only'] = eda_df['date_normalized'].dt.date
            
            # Group by date and count
            email_counts = eda_df.groupby('date_only').size()
            
            if not email_counts.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                email_counts.plot(kind='line', ax=ax)
                plt.title('Email Volume Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Emails')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No valid date data for volume chart.")
        except Exception as e:
            st.error(f"Error creating email volume chart: {str(e)}")
    
    # 2. Top Senders Analysis
    st.subheader("Top Email Senders")
    if 'from' in eda_df.columns:
        try:
            # Extract email addresses from 'from' field
            eda_df['sender_email'] = eda_df['from'].apply(
                lambda x: re.search(r'<([^>]+)>', x).group(1) if isinstance(x, str) and re.search(r'<([^>]+)>', x) else x
            )
            
            # Count occurrences of each sender
            sender_counts = eda_df['sender_email'].value_counts()
            top_senders = sender_counts.head(10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            top_senders.plot(kind='bar', ax=ax)
            plt.title('Top 10 Email Senders')
            plt.xlabel('Sender')
            plt.ylabel('Number of Emails')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating top senders chart: {str(e)}")
    
    # 3. Word Frequency Analysis
    st.subheader("Common Words in Emails")
    if 'clean_body' in eda_df.columns:
        try:
            # Combine all email bodies
            all_text = ' '.join(eda_df['clean_body'].dropna())
            
            if all_text.strip():  # Check if there's actual text to analyze
                # Create word cloud
                stop_words = set(stopwords.words('english'))
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    stopwords=stop_words,
                                    max_words=100).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No text content for word cloud generation.")
        except Exception as e:
            st.error(f"Error creating word cloud: {str(e)}")
    
    # 4. Email Length Distribution
    st.subheader("Email Length Distribution")
    if 'body' in eda_df.columns:
        try:
            eda_df['body_length'] = eda_df['body'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.hist(eda_df['body_length'].clip(upper=500), bins=50)  # Clip to 500 words for better visualization
            plt.title('Email Length Distribution')
            plt.xlabel('Number of Words')
            plt.ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating email length chart: {str(e)}")
    
    # 5. Topic Modeling
    st.subheader("Email Topics")
    if 'clean_body' in eda_df.columns:
        try:
            # Filter out empty bodies
            non_empty_bodies = eda_df['clean_body'].dropna().tolist()
            
            if non_empty_bodies:
                # Vectorize the text
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
                tf = vectorizer.fit_transform(non_empty_bodies)
                
                # Fit LDA
                lda = LatentDirichletAllocation(n_components=5, random_state=42)
                lda.fit(tf)
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Print top 10 words for each topic
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[:-11:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    st.write(f"**Topic {topic_idx+1}:** {', '.join(top_words)}")
            else:
                st.warning("No content for topic modeling.")
        except Exception as e:
            st.error(f"Error performing topic modeling: {str(e)}")
    
    # Return the sender counts if available, otherwise return None
    return {'sender_counts': sender_counts}