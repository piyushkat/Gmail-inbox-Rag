import base64
import datetime
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
from email.utils import parsedate_to_datetime
import re

def get_emails(service, progress_bar=None, after_date=None):
    """
    Fetch emails from Gmail, optionally only those after a specific date.
    
    Args:
        service: Gmail API service instance
        progress_bar: Streamlit progress bar
        after_date: If provided, only fetch emails after this date (RFC 3339 timestamp)
    
    Returns:
        List of email data dictionaries
    """
    query = ""
    if after_date:
        # Convert datetime to Gmail query format
        if isinstance(after_date, datetime.datetime):
            query = f"after:{after_date.strftime('%Y/%m/%d')}"
            st.write(f"Fetching emails after {after_date.strftime('%Y-%m-%d %H:%M:%S')}...")
        else:
            st.write(f"Fetching emails with query: {after_date}")
            query = after_date
    else:
        st.write("Fetching all emails...")
    
    # Get list of email IDs
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])
    
    # Continue fetching all emails using nextPageToken
    next_page_token = results.get('nextPageToken')
    while next_page_token:
        results = service.users().messages().list(userId='me', q=query, pageToken=next_page_token).execute()
        messages.extend(results.get('messages', []))
        next_page_token = results.get('nextPageToken')
    
    if not messages:
        if after_date:
            st.info("No new messages found since last check.")
        else:
            st.warning("No messages found.")
        return []
    
    emails_data = []
    
    if progress_bar:
        progress_bar.progress(0, text="Processing emails...")
    
    for i, message in enumerate(messages):
        if progress_bar:
            progress_bar.progress((i + 1) / len(messages), text=f"Processing email {i+1}/{len(messages)}")
        
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        
        # Get email headers
        headers = msg['payload'].get('headers', [])
        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
        from_email = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'Unknown')
        date_str = next((header['value'] for header in headers if header['name'].lower() == 'date'), None)
        
        # Parse date with improved handling
        if date_str:
            try:
                # Try using email.utils parser first (handles many formats)
                date_obj = parsedate_to_datetime(date_str)
            except:
                try:
                    # Fallback to manual parsing
                    date_str = date_str.split('(')[0].strip()
                    date_obj = datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
                except:
                    try:
                        # Try without timezone
                        date_obj = datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S')
                    except:
                        date_obj = None
        else:
            date_obj = None
        
        # Get message body
        body = ""
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                    break
                elif part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                    html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                    body = BeautifulSoup(html_body, 'html.parser').get_text(separator=' ', strip=True)
                    break
        elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
            if msg['payload'].get('mimeType') == 'text/plain':
                body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8', errors='replace')
            elif msg['payload'].get('mimeType') == 'text/html':
                html_body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8', errors='replace')
                body = BeautifulSoup(html_body, 'html.parser').get_text(separator=' ', strip=True)
        
        # Extract labels/categories
        labels = msg.get('labelIds', [])
        
        # Store email data
        email_data = {
            'id': message['id'],
            'subject': subject,
            'from': from_email,
            'date': date_obj,
            'body': body,
            'labels': labels
        }
        
        emails_data.append(email_data)
    
    st.success(f"Successfully fetched {len(emails_data)} emails.")
    return emails_data

def fetch_new_emails():
    """
    Function to fetch new emails since the last check.
    This will be called by the scheduler.
    """
    if not st.session_state.get('scheduler_running', False):
        return
        
    if not st.session_state.get('authenticated', False) or not st.session_state.get('service'):
        print("Not authenticated, skipping scheduled email fetch")
        return
        
    print(f"[{datetime.datetime.now()}] Checking for new emails...")
    
    # Get the last email date we have
    last_email_date = st.session_state.get('last_email_date')
    
    # Fetch new emails
    new_emails = get_emails(st.session_state.service, after_date=last_email_date)
    
    if new_emails:
        print(f"Found {len(new_emails)} new emails")
        
        # Convert to DataFrame
        new_emails_df = pd.DataFrame(new_emails)
        new_emails_df['clean_body'] = new_emails_df['body'].apply(clean_text)
        
        # Update session state DataFrame
        if st.session_state.emails_df is not None:
            # Combine old and new emails, avoiding duplicates by ID
            combined_df = pd.concat([st.session_state.emails_df, new_emails_df])
            st.session_state.emails_df = combined_df.drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
        else:
            st.session_state.emails_df = new_emails_df
        
        # Update the vector database with new emails
        if st.session_state.get('vector_db') and hasattr(st.session_state.vector_db, 'collection_created'):
            success = update_vector_database(new_emails_df, st.session_state.vector_db)
            if success:
                print("Vector database updated with new emails")
            else:
                print("Failed to update vector database")
        
        # Update the last email date
        if len(new_emails_df) > 0 and 'date' in new_emails_df.columns:
            max_date = pd.to_datetime(new_emails_df['date'], errors='coerce').max()
            if pd.notna(max_date):
                st.session_state['last_email_date'] = max_date
                print(f"Updated last email date to {max_date}")
    else:
        print("No new emails found")