import os
import pickle
import json
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import streamlit as st

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate with Gmail API and return the service."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load client secrets from file
            client_config = json.load(open('client_secret_346950082405-8n5q1bapmbpmq9oravterf9mkuf0md14.apps.googleusercontent.com.json'))
            
            # Set up the OAuth flow for a web application
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                redirect_uri='http://localhost:8080'
            )
            
            # Generate the authorization URL
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            st.info(f"Please go to this URL and authorize the application: {auth_url}")
            
            # Get the authorization code from the user
            auth_code = st.text_input("Enter the authorization code:")
            
            if auth_code:
                # Exchange the authorization code for credentials
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                
                # Save the credentials for the next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
                
                service = build('gmail', 'v1', credentials=creds)
                return service
            else:
                return None
        
    service = build('gmail', 'v1', credentials=creds)
    return service