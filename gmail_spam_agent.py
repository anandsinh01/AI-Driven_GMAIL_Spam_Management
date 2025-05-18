"""
Gmail Spam Management Agent

This application uses the Gmail API to identify and delete spam emails based on 
custom rules and machine learning classification.
"""

import os
import pickle
import base64
import re
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Gmail API imports
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ML libraries for spam detection
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gmail_spam_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GmailSpamAgent")

class GmailSpamAgent:
    """Agent for managing spam in Gmail using MCP approach."""
    
    # Gmail API scopes
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    
    def __init__(self, credentials_path: str = 'credentials.json',
                 token_path: str = 'token.pickle'):
        """
        Initialize the Gmail Spam Agent.

        Args:
            credentials_path (str): Path to the credentials.json file (default: 'credentials.json')
            token_path (str): Path to save/load the authentication token (default: 'token.pickle')
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.classifier = None
        self.vectorizer = None
        
        # MCP rules configuration
        self.rules = {
            "obvious_spam_keywords": [
                "lottery", "winner", "million dollars", "nigerian prince", 
                "claim your prize", "urgent money transfer", "bitcoin opportunity",
                "investment opportunity", "pharmaceutical", "miracle cure", 
                "enlargement", "weight loss guarantee", "work from home scheme",
                "make money fast", "earn extra cash", "MLM opportunity"
            ],
            "suspicious_email_domains": [
                "xyz123mail.com", "freemail4u.net", "supercheapdomains.co", 
                "scamalert.org", "temporarymail.org", "disposableemail.com"
            ],
            "max_spam_score": 0.85,  # Threshold for ML classifier
            "min_confidence": 0.75,  # Minimum confidence to auto-delete
            "max_emails_per_run": 100,  # Maximum emails to process per run
            "ignore_contacts": True,  # Ignore emails from contacts
            "age_threshold_days": 30,  # Don't process emails older than this
            "learning_mode": False,  # If True, don't delete but label for training
        }
        
    def authenticate(self) -> bool:
        """Authenticate with Gmail API."""
        creds = None
        
        # Load token if it exists
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)

        # Refresh token if needed
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing token: {e}")
                creds = None

        # If no valid creds, need to authenticate
        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)

                # Save the credentials
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return False

        # Build the service
        try:
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Successfully authenticated with Gmail API")
            return True
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {e}")
            return False
    
    def train_classifier(self, training_data_path: Optional[str] = None) -> bool:
        """
        Train the spam classifier.
        
        Args:
            training_data_path: Path to CSV with 'text' and 'label' columns
                               (if None, fetch from Gmail)
        
        Returns:
            bool: True if training was successful
        """
        try:
            if training_data_path and os.path.exists(training_data_path):
                # Load training data from file
                df = pd.read_csv(training_data_path)
            else:
                # Fetch training data from Gmail
                df = self._fetch_training_data()
                
            if df is None or len(df) < 10:
                logger.warning("Insufficient training data")
                return False
                
            # Prepare data
            X = df['text']
            y = df['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create and train vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.7,
                stop_words='english'
            )
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            
            # Train classifier
            self.classifier = MultinomialNB()
            self.classifier.fit(X_train_vectorized, y_train)
            
            # Evaluate
            X_test_vectorized = self.vectorizer.transform(X_test)
            accuracy = self.classifier.score(X_test_vectorized, y_test)
            logger.info(f"Classifier trained with accuracy: {accuracy:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            return False
    
    def _fetch_training_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch training data from Gmail.
        
        Returns:
            DataFrame with 'text' and 'label' columns or None if failed
        """
        if not self.service:
            logger.error("Gmail service not initialized")
            return None
            
        try:
            # Get spam emails (from spam folder)
            spam_emails = self._get_emails_from_folder('SPAM', max_results=500)
            
            # Get ham emails (from inbox)
            ham_emails = self._get_emails_from_folder('INBOX', max_results=500)
            
            # Create DataFrame
            texts = []
            labels = []
            
            for email in spam_emails:
                text = self._extract_email_text(email)
                if text:
                    texts.append(text)
                    labels.append(1)  # 1 for spam
                    
            for email in ham_emails:
                text = self._extract_email_text(email)
                if text:
                    texts.append(text)
                    labels.append(0)  # 0 for ham
            
            df = pd.DataFrame({
                'text': texts,
                'label': labels
            })
            
            logger.info(f"Fetched training data: {len(df)} emails " +
                       f"({len(df[df['label'] == 1])} spam, {len(df[df['label'] == 0])} ham)")
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch training data: {e}")
            return None
    
    def _get_emails_from_folder(self, folder: str, max_results: int = 100) -> List[Dict]:
        """
        Get emails from a specific folder.
        
        Args:
            folder: Folder name (INBOX, SPAM, etc.)
            max_results: Maximum number of emails to retrieve
            
        Returns:
            List of email objects
        """
        query = f"in:{folder}"
        
        # Add date filter to avoid older emails
        cutoff_date = (datetime.now() - timedelta(days=self.rules['age_threshold_days'])).strftime('%Y/%m/%d')
        query += f" after:{cutoff_date}"
        
        emails = []
        try:
            # Get message IDs
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            # Get actual message content
            for msg in messages:
                try:
                    msg_full = self.service.users().messages().get(
                        userId='me', id=msg['id']
                    ).execute()
                    emails.append(msg_full)
                except Exception as e:
                    logger.error(f"Error fetching message {msg['id']}: {e}")
                    continue
                    
            return emails
        except Exception as e:
            logger.error(f"Error fetching emails from {folder}: {e}")
            return []
    
    def _extract_email_text(self, email: Dict) -> Optional[str]:
        """
        Extract text content from email.
        
        Args:
            email: Gmail API email object
            
        Returns:
            String with email text or None if extraction failed
        """
        try:
            payload = email.get('payload', {})
            parts = payload.get('parts', [])
            headers = payload.get('headers', [])
            
            # Extract subject
            subject = ""
            for header in headers:
                if header['name'].lower() == 'subject':
                    subject = header['value']
                    break
            
            # Extract body text
            body = ""
            
            # Check for plain text in the main payload
            if 'body' in payload and 'data' in payload['body']:
                body_data = payload['body']['data']
                body = base64.urlsafe_b64decode(body_data).decode('utf-8')
            
            # If not found, check parts
            if not body and parts:
                for part in parts:
                    if part['mimeType'] == 'text/plain' and 'body' in part and 'data' in part['body']:
                        body_data = part['body']['data']
                        body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                        break
            
            # Combine subject and body
            full_text = f"{subject}\n\n{body}"
            
            # Remove HTML tags if any
            text_without_tags = re.sub(r'<[^>]+>', ' ', full_text)
            
            # Normalize whitespace
            normalized_text = re.sub(r'\s+', ' ', text_without_tags).strip()
            
            return normalized_text
        except Exception as e:
            logger.error(f"Error extracting email text: {e}")
            return None
    
    def scan_inbox(self) -> List[Dict]:
        """
        Scan inbox for spam emails.
        
        Returns:
            List of potential spam email objects
        """
        if not self.service:
            logger.error("Gmail service not initialized")
            return []
        
        try:
            # Get unread emails from inbox
            query = "in:inbox"
            
            # Add date filter
            cutoff_date = (datetime.now() - timedelta(days=self.rules['age_threshold_days'])).strftime('%Y/%m/%d')
            query += f" after:{cutoff_date}"
            
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=self.rules['max_emails_per_run']
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                logger.info("No messages found in inbox")
                return []
            
            logger.info(f"Found {len(messages)} messages in inbox for scanning")
            
            potential_spam = []
            for msg in messages:
                try:
                    # Get full message
                    msg_full = self.service.users().messages().get(
                        userId='me', id=msg['id']
                    ).execute()
                    
                    # Check if it's spam
                    is_spam, confidence, reasons = self._classify_email(msg_full)
                    
                    if is_spam:
                        potential_spam.append({
                            'message': msg_full,
                            'confidence': confidence,
                            'reasons': reasons
                        })
                except Exception as e:
                    logger.error(f"Error processing message {msg['id']}: {e}")
                    continue
            
            logger.info(f"Identified {len(potential_spam)} potential spam messages")
            return potential_spam
        except Exception as e:
            logger.error(f"Error scanning inbox: {e}")
            return []
    
    def _classify_email(self, email: Dict) -> Tuple[bool, float, List[str]]:
        """
        Classify an email as spam or not.
        
        Args:
            email: Gmail API email object
            
        Returns:
            Tuple of (is_spam, confidence, reasons)
        """
        reasons = []
        sender = self._get_sender(email)
        subject = self._get_subject(email)
        body_text = self._extract_email_text(email) or ""
        
        # Rule 1: Check if sender is in contacts (if enabled)
        if self.rules['ignore_contacts'] and self._is_in_contacts(sender):
            return False, 0.0, ["Sender is in contacts"]
        
        # Rule 2: Check for suspicious domains
        for domain in self.rules['suspicious_email_domains']:
            if domain in sender.lower():
                reasons.append(f"Suspicious domain: {domain}")
        
        # Rule 3: Check for spam keywords in subject and body
        for keyword in self.rules['obvious_spam_keywords']:
            if keyword.lower() in subject.lower() or keyword.lower() in body_text.lower():
                reasons.append(f"Spam keyword: {keyword}")
        
        # Rule 4: Use ML classifier if available
        ml_score = 0.0
        if self.classifier and self.vectorizer and body_text:
            try:
                # Vectorize the text
                text_vectorized = self.vectorizer.transform([body_text])
                
                # Predict probability
                spam_prob = self.classifier.predict_proba(text_vectorized)[0][1]
                ml_score = spam_prob
                
                if spam_prob >= self.rules['max_spam_score']:
                    reasons.append(f"ML classifier score: {spam_prob:.4f}")
            except Exception as e:
                logger.error(f"ML classification error: {e}")
        
        # Calculate overall confidence based on rules and ML
        # More reasons = higher confidence
        base_confidence = min(0.3 + (len(reasons) * 0.15), 0.9)
        
        # Incorporate ML score if available
        if ml_score > 0:
            confidence = (base_confidence + ml_score) / 2
        else:
            confidence = base_confidence
            
        # Determine if it's spam based on confidence threshold
        is_spam = confidence >= self.rules['min_confidence']
        
        return is_spam, confidence, reasons
    
    def _get_sender(self, email: Dict) -> str:
        """Extract sender from email."""
        headers = email.get('payload', {}).get('headers', [])
        for header in headers:
            if header['name'].lower() in ('from', 'sender'):
                return header['value']
        return ""
    
    def _get_subject(self, email: Dict) -> str:
        """Extract subject from email."""
        headers = email.get('payload', {}).get('headers', [])
        for header in headers:
            if header['name'].lower() == 'subject':
                return header['value']
        return ""
    
    def _is_in_contacts(self, sender: str) -> bool:
        """Check if sender is in contacts."""
        # Extract email from sender string (Name <email@example.com>)
        email_match = re.search(r'<([^>]+)>', sender)
        email = email_match.group(1) if email_match else sender
        
        try:
            # Use People API to check contacts
            # This is simplified; in reality, you would need to implement proper
            # contact checking using the People API
            return False  # Placeholder
        except Exception:
            return False
    
    def delete_spam(self, potential_spam: List[Dict]) -> int:
        """
        Delete identified spam emails.
        
        Args:
            potential_spam: List of potential spam email objects
            
        Returns:
            Number of deleted emails
        """
        if not self.service:
            logger.error("Gmail service not initialized")
            return 0
            
        if not potential_spam:
            logger.info("No spam to delete")
            return 0
            
        deleted_count = 0
        
        for spam in potential_spam:
            msg = spam['message']
            confidence = spam['confidence']
            reasons = spam['reasons']
            
            # Only delete if confidence is high enough
            if confidence >= self.rules['min_confidence']:
                try:
                    if self.rules['learning_mode']:
                        # In learning mode, just label as spam
                        self.service.users().messages().modify(
                            userId='me',
                            id=msg['id'],
                            body={'addLabelIds': ['SPAM']}
                        ).execute()
                        logger.info(f"Marked as spam (learning mode): {msg['id']}")
                    else:
                        # Actually delete (trash) the message
                        self.service.users().messages().trash(
                            userId='me',
                            id=msg['id']
                        ).execute()
                        logger.info(f"Deleted spam: {msg['id']}")
                    
                    deleted_count += 1
                    
                    # Log details
                    subject = self._get_subject(msg)
                    sender = self._get_sender(msg)
                    logger.info(f"Processed spam from {sender}, subject: {subject}")
                    logger.info(f"Reasons: {', '.join(reasons)}")
                except Exception as e:
                    logger.error(f"Error deleting message {msg['id']}: {e}")
            else:
                logger.info(f"Confidence too low ({confidence:.4f}) to delete message {msg['id']}")
        
        return deleted_count
    
    def run(self, learning_mode: bool = False, delete: bool = True) -> Dict[str, Any]:
        """
        Run the spam detection and removal process.
        
        Args:
            learning_mode: If True, only label spam but don't delete
            delete: If True, delete identified spam
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Update learning mode setting
        self.rules['learning_mode'] = learning_mode
        
        # Initialize results
        results = {
            'success': False,
            'scanned_count': 0,
            'spam_detected': 0,
            'spam_deleted': 0,
            'execution_time': 0,
            'errors': []
        }
        
        # Authenticate with Gmail API
        if not self.authenticate():
            results['errors'].append("Authentication failed")
            return results
        
        # Train classifier if not already trained
        if not self.classifier:
            if not self.train_classifier():
                results['errors'].append("Classifier training failed, using rule-based detection only")
        
        # Scan inbox for spam
        potential_spam = self.scan_inbox()
        results['scanned_count'] = self.rules['max_emails_per_run']
        results['spam_detected'] = len(potential_spam)
        
        # Delete spam if requested
        if delete and potential_spam:
            results['spam_deleted'] = self.delete_spam(potential_spam)
        
        # Calculate execution time
        results['execution_time'] = time.time() - start_time
        results['success'] = True
        
        logger.info(f"Scan completed in {results['execution_time']:.2f} seconds")
        logger.info(f"Detected {results['spam_detected']} spam emails")
        logger.info(f"Deleted {results['spam_deleted']} spam emails")
        
        return results

    def delete_unread_before_last_month(self) -> int:
        """
        Delete all unread emails received before the first day of the current month.
        Returns:
            Number of deleted emails
        """
        if not self.service:
            logger.error("Gmail service not initialized")
            return 0

        # Calculate the first day of the current month
        now = datetime.now()
        first_of_month = datetime(now.year, now.month, 1)
        cutoff_date = first_of_month.strftime('%Y/%m/%d')

        # Gmail search query for unread emails before this month
        query = f"is:unread before:{cutoff_date}"
        deleted_count = 0
        try:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=500
            ).execute()
            messages = results.get('messages', [])
            if not messages:
                logger.info("No unread emails found before this month.")
                return 0
            for msg in messages:
                try:
                    self.service.users().messages().trash(
                        userId='me',
                        id=msg['id']
                    ).execute()
                    deleted_count += 1
                    logger.info(f"Deleted unread email: {msg['id']}")
                except Exception as e:
                    logger.error(f"Error deleting unread email {msg['id']}: {e}")
            logger.info(f"Deleted {deleted_count} unread emails before {cutoff_date}.")
            return deleted_count
        except Exception as e:
            logger.error(f"Error searching for unread emails: {e}")
            return 0

    def archive_unread_before_date(self, cutoff_date: Optional[str] = None) -> int:
        """
        Move all unread emails received before the given cutoff_date to a 'MyArchive' label.
        Args:
            cutoff_date: Date string in 'YYYY/MM/DD' format. If None, uses first of current month.
        Returns:
            Number of archived emails
        """
        if not self.service:
            logger.error("Gmail service not initialized")
            return 0

        # Default to first of current month if not provided
        if not cutoff_date:
            now = datetime.now()
            first_of_month = datetime(now.year, now.month, 1)
            cutoff_date = first_of_month.strftime('%Y/%m/%d')

        # Gmail search query for unread emails before the cutoff date
        query = f"is:unread before:{cutoff_date}"
        logger.info(f"Gmail API query: {query}")
        archived_count = 0
        archive_label_id = None
        archive_label_name = 'MyArchive'
        total_found = 0
        first_message_ids = []

        # Step 1: Check if 'MyArchive' label exists, else create it
        try:
            labels_result = self.service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            for label in labels:
                if label['name'].lower() == archive_label_name.lower():
                    archive_label_id = label['id']
                    break
            if not archive_label_id:
                label_obj = {
                    'name': archive_label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                new_label = self.service.users().labels().create(userId='me', body=label_obj).execute()
                archive_label_id = new_label['id']
                logger.info(f"Created '{archive_label_name}' label.")
        except Exception as e:
            logger.error(f"Error checking/creating {archive_label_name} label: {e}")
            return 0

        # Step 2: Find unread emails before the cutoff date and move them to MyArchive (with pagination)
        try:
            next_page_token = None
            page = 1
            while True:
                results = self.service.users().messages().list(
                    userId='me', q=query, maxResults=500, pageToken=next_page_token
                ).execute()
                messages = results.get('messages', [])
                logger.info(f"Page {page}: Found {len(messages)} unread emails before {cutoff_date}.")
                if messages:
                    logger.info(f"First 5 message IDs on page {page}: {[msg['id'] for msg in messages[:5]]}")
                if page == 1 and messages:
                    first_message_ids = [msg['id'] for msg in messages[:3]]
                total_found += len(messages)
                if not messages:
                    break
                for msg in messages:
                    try:
                        self.service.users().messages().modify(
                            userId='me',
                            id=msg['id'],
                            body={'addLabelIds': [archive_label_id], 'removeLabelIds': ['INBOX']}
                        ).execute()
                        archived_count += 1
                        logger.info(f"Archived unread email: {msg['id']}")
                    except Exception as e:
                        logger.error(f"Error archiving unread email {msg['id']}: {e}")
                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break
                page += 1
            logger.info(f"Total unread emails found by API: {total_found}")
            # Fetch and log subject/date for the first 3 emails
            for msg_id in first_message_ids:
                try:
                    msg_full = self.service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['Subject', 'Date']).execute()
                    headers = msg_full.get('payload', {}).get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '(no subject)')
                    date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '(no date)')
                    logger.info(f"Sample message: ID={msg_id}, Subject={subject}, Date={date}")
                except Exception as e:
                    logger.error(f"Error fetching details for message {msg_id}: {e}")
            logger.info(f"Archived {archived_count} unread emails before {cutoff_date} to '{archive_label_name}'.")
            return archived_count
        except Exception as e:
            logger.error(f"Error searching for unread emails: {e}")
            return 0

    def fetch_emails_after_date(self, date_str):
        """
        Fetch emails received after the given date (YYYY/MM/DD).
        Returns a list of dicts with at least 'id', 'from', and 'subject' keys.
        """
        if not self.service:
            return []
        query = f"after:{date_str.replace('/', '-')}"
        try:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=50
            ).execute()
            messages = results.get('messages', [])
            emails = []
            for msg in messages:
                msg_full = self.service.users().messages().get(
                    userId='me', id=msg['id']
                ).execute()
                sender = self._get_sender(msg_full)
                subject = self._get_subject(msg_full)
                emails.append({'id': msg['id'], 'from': sender, 'subject': subject})
            return emails
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []

    def fetch_all_unread_emails(self, max_to_fetch=None):
        """
        Fetch all unread emails (optionally up to max_to_fetch).
        Returns a list of dicts with 'id', 'from', and 'subject' keys.
        """
        if not self.service:
            return []
        query = "is:unread"
        emails = []
        next_page_token = None
        try:
            while True:
                results = self.service.users().messages().list(
                    userId='me', q=query, maxResults=500, pageToken=next_page_token
                ).execute()
                messages = results.get('messages', [])
                for msg in messages:
                    msg_full = self.service.users().messages().get(
                        userId='me', id=msg['id']
                    ).execute()
                    sender = self._get_sender(msg_full)
                    subject = self._get_subject(msg_full)
                    emails.append({'id': msg['id'], 'from': sender, 'subject': subject})
                    if max_to_fetch and len(emails) >= max_to_fetch:
                        return emails
                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break
            return emails
        except Exception as e:
            print(f"Error fetching all unread emails: {e}")
            return []

    def delete_emails(self, emails):
        """
        Delete a list of emails by their 'id'.
        Returns the number of deleted emails.
        """
        if not self.service or not emails:
            return 0
        deleted_count = 0
        for email in emails:
            try:
                self.service.users().messages().trash(
                    userId='me',
                    id=email['id']
                ).execute()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting email: {e}")
        return deleted_count

    def move_emails_to_label(self, emails, label_name):
        """
        Move the given emails to the specified label (create label if needed).
        """
        if not self.service or not emails:
            return 0
        # Check/create label
        try:
            labels_result = self.service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            label_id = None
            for label in labels:
                if label['name'].lower() == label_name.lower():
                    label_id = label['id']
                    break
            if not label_id:
                label_obj = {
                    'name': label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                new_label = self.service.users().labels().create(userId='me', body=label_obj).execute()
                label_id = new_label['id']
        except Exception as e:
            print(f"Error checking/creating label: {e}")
            return 0

        # Move emails
        moved_count = 0
        for email in emails:
            try:
                self.service.users().messages().modify(
                    userId='me',
                    id=email['id'],
                    body={'addLabelIds': [label_id], 'removeLabelIds': ['INBOX']}
                ).execute()
                moved_count += 1
            except Exception as e:
                print(f"Error moving email: {e}")
        return moved_count

    def process_natural_language_query(self, query: str, context: str = "") -> str:
        """
        Process a natural language query using OpenAI GPT-4-Turbo.
        Optionally provide context (such as email summaries).
        """
        import openai
        import streamlit as st
        openai.api_key = st.session_state.get('openai_api_key', None)
        system_prompt = (
            "You are an advanced email assistant that helps users manage their Gmail. "
            "You can help with tasks like finding suspicious emails, reading and categorizing specific types of emails, "
            "analyzing email patterns, summarizing email content, and providing recommendations for email management. "
            "Please provide helpful, concise, and actionable responses."
        )
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": query})
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing your request: {e}"

    def get_unread_count(self):
        """
        Get the total number of unread emails in the inbox using label metadata.
        """
        if not self.service:
            return 0
        try:
            results = self.service.users().labels().get(userId='me', id='INBOX').execute()
            return results.get('messagesUnread', 0)
        except Exception as e:
            print(f"Error getting unread count: {e}")
            return 0

def main():
    """Main function to run the Gmail Spam Agent."""
    print("Gmail Spam Management Agent - Starting")
    print("======================================")
    
    # Create agent
    agent = GmailSpamAgent()
    
    print("\nOptions:")
    print("1. Run spam scan (learning mode, no deletion)")
    print("2. Run spam scan and delete spam")
    print("3. Delete all unread emails before this month")
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nStep 1: Running in learning mode (no deletion)...")
        results = agent.run(learning_mode=True, delete=False)
        if results['success']:
            print(f"✓ Learning mode scan completed successfully")
            print(f"✓ Detected {results['spam_detected']} potential spam emails")
            print(f"✓ These emails were labeled as spam but not deleted")
        else:
            print("✗ Learning mode scan failed")
            for error in results['errors']:
                print(f"  - {error}")
    elif choice == '2':
        print("\nStep 2: Running in delete mode...")
        results = agent.run(learning_mode=False, delete=True)
        if results['success']:
            print(f"✓ Delete mode scan completed successfully")
            print(f"✓ Detected {results['spam_detected']} spam emails")
            print(f"✓ Deleted {results['spam_deleted']} spam emails")
        else:
            print("✗ Delete mode scan failed")
            for error in results['errors']:
                print(f"  - {error}")
    elif choice == '3':
        print("\nDeleting all unread emails before this month...")
        agent.authenticate()
        deleted_count = agent.delete_unread_before_last_month()
        print(f"✓ Deleted {deleted_count} unread emails before this month.")
    else:
        print("Invalid choice. Exiting.")
    print("\nDone! Check the log file for more details.")

if __name__ == "__main__":
    main()
