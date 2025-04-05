import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd

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