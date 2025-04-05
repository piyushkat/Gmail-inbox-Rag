import base64
import datetime
from bs4 import BeautifulSoup
import streamlit as st
from email.utils import parsedate_to_datetime

def get_emails(service, progress_bar=None):
    """Fetch all emails from Gmail."""
    st.write("Fetching all emails...")
    
    # Get list of email IDs
    results = service.users().messages().list(userId='me').execute()
    messages = results.get('messages', [])
    
    # Continue fetching all emails using nextPageToken
    next_page_token = results.get('nextPageToken')
    while next_page_token:
        results = service.users().messages().list(userId='me', pageToken=next_page_token).execute()
        messages.extend(results.get('messages', []))
        next_page_token = results.get('nextPageToken')
    
    if not messages:
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