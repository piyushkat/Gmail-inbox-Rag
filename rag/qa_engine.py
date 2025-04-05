import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_embedding_model():
    """Initialize and return the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def ask_question_with_groq(question, client):
    """Query the vector database with a question and generate an answer using Groq API."""
    with st.spinner("Searching emails and generating answer..."):
        embed_model = get_embedding_model()
        
        # Use the provided client to get the collection
        collection = client.get_collection("email_collection")
        
        question_embedding = embed_model.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=8  # Increased from 5 to 8 for better context
        )
        
        relevant_docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context = "\n\n---\n\n".join([
            f"Email {i+1}:\nFrom: {meta['from']}\nDate: {meta['date']}\nSubject: {meta['subject']}\n\n{doc}"
            for i, (doc, meta) in enumerate(zip(relevant_docs, metadatas))
        ])
        
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Improved prompt for more accurate and focused answers
            prompt = f"""
            You are an email assistant that provides precise, accurate answers to questions about emails.
            
            Analyze the following emails from the user's inbox to answer their question: "{question}"
            
            {context}
            
            Important instructions:
            1. Only use information clearly present in the provided emails
            2. If the emails don't contain information to answer the question, say "I don't have enough information in the provided emails to answer this question"
            3. Focus on the user's specific question and provide a direct, concise answer
            4. Include key details from the emails that support your answer
            5. If multiple emails contain relevant information, synthesize them into a coherent answer
            6. Don't make assumptions about content not shown in the emails
            7. If the question is ambiguous, address the most likely interpretation based on the emails
            
            Give a precise, helpful answer based solely on the emails provided.
            """
            
            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about emails based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,  # Increased from 500 to 800 for more detailed answers
                "temperature": 0.3  # Lower temperature for more factual responses
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
            else:
                answer = f"Error with Groq API: {response.status_code} - {response.text}"
        except Exception as e:
            answer = f"Error generating answer: {str(e)}\n\nHere's the most relevant email content for your question:\n\n{relevant_docs[0]}"
    
    return {
        "answer": answer,
        "sources": [{"metadata": meta, "excerpt": doc[:200] + "..."} for doc, meta in zip(relevant_docs, metadatas)]
    }
