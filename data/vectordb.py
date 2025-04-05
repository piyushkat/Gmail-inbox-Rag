import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

def initialize_vector_db():
    """Initialize and return a ChromaDB client."""
    try:
        os.makedirs("./chroma_db", exist_ok=True)
        db_client = chromadb.PersistentClient(path="./chroma_db")
        return db_client, True
    except Exception as e:
        st.error(f"Failed to initialize Chroma client: {str(e)}")
        return None, False

def create_vector_database(emails_df, client, progress_bar=None):
    """Create and populate a vector database from email data."""
    st.subheader("Creating vector database...")
    
    # Initialize embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create or get the collection using the provided client
    try:
        # First try to delete the collection if it exists
        try:
            client.delete_collection("email_collection")
            st.info("Deleted existing collection.")
        except:
            pass
        
        # Create a new collection
        collection = client.create_collection(
            name="email_collection",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        st.info("Created new email collection.")
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return False
    
    # Prepare data for insertion
    ids = emails_df['id'].tolist()
    documents = []
    metadatas = []
    
    for _, row in emails_df.iterrows():
        subject = row['subject'] if isinstance(row['subject'], str) else ''
        body = row['body'] if isinstance(row['body'], str) else ''
        full_text = f"Subject: {subject}\n\nBody: {body}"
        
        metadata = {
            'subject': subject[:100],
            'from': row['from'][:100] if isinstance(row['from'], str) else '',
            'date': str(row['date']) if pd.notna(row['date']) else ''
        }
        
        documents.append(full_text)
        metadatas.append(metadata)
    
    batch_size = 32
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    if progress_bar:
        progress_bar.progress(0, text="Creating vector database...")
    
    for i in range(0, len(documents), batch_size):
        if progress_bar:
            batch_num = i // batch_size + 1
            progress_bar.progress(batch_num / total_batches, text=f"Processing batch {batch_num}/{total_batches}")
        
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        batch_embeddings = embed_model.encode(batch_docs).tolist()
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta
        )
    
    st.success(f"Vector database created with {len(ids)} email entries")
    return True  # Return success status

def update_vector_database(new_emails_df, client):
    """Update the existing vector database with new emails."""
    if new_emails_df is None or len(new_emails_df) == 0:
        return False
    
    try:
        # Initialize embedding model
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get the collection
        collection = client.get_collection("email_collection")
        
        # Prepare data for insertion
        ids = new_emails_df['id'].tolist()
        documents = []
        metadatas = []
        
        for _, row in new_emails_df.iterrows():
            subject = row['subject'] if isinstance(row['subject'], str) else ''
            body = row['body'] if isinstance(row['body'], str) else ''
            full_text = f"Subject: {subject}\n\nBody: {body}"
            
            metadata = {
                'subject': subject[:100],
                'from': row['from'][:100] if isinstance(row['from'], str) else '',
                'date': str(row['date']) if pd.notna(row['date']) else ''
            }
            
            documents.append(full_text)
            metadatas.append(metadata)
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            batch_embeddings = embed_model.encode(batch_docs).tolist()
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_meta
            )
        
        return True
    except Exception as e:
        print(f"Error updating vector database: {str(e)}")
        return False

def query_vector_database(question, client, embed_model):
    """Query the vector database for relevant email content."""
    collection = client.get_collection("email_collection")
    
    question_embedding = embed_model.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=8  # Increased from 5 to 8 for better context
    )
    
    return {
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0]
    }