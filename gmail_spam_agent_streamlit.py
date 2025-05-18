import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import base64
from io import BytesIO
from datetime import datetime, timedelta
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import requests
import openai
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="Gmail Spam Agent",
    page_icon="📧",
    layout="wide"
)

# --- Modern Button Styling ---
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4285F4;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        font-weight: 600;
        margin: 0.5em 0.2em 0.5em 0;
        cursor: pointer;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background-color: #174ea6;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Gmail API imports
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Import the core spam agent functionality
from gmail_spam_agent import GmailSpamAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gmail_spam_agent_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GmailSpamAgentUI")

# Initialize session state variables
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'spam_emails' not in st.session_state:
    st.session_state.spam_emails = []
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'history' not in st.session_state:
    st.session_state.history = []

# Conversational Agent (Gemini LLM) UI and logic
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def add_log(message, level="info"):
    """Add a log message to session state with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)

def display_header():
    """Display the app header."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Gmail_icon_%282020%29.svg/512px-Gmail_icon_%282020%29.svg.png", width=80)
    with col2:
        st.title("Gmail Spam Management Agent")
        st.markdown("An intelligent agent to detect and remove spam emails using machine learning")

def setup_sidebar():
    """Set up the sidebar with authentication and controls."""
    with st.sidebar:
        st.header("Controls")
        
        # Authentication section
        st.subheader("Authentication")
        if not st.session_state.authenticated:
            st.info("Please authenticate with your Google account to proceed.")
            
            # File uploader for credentials
            uploaded_file = st.file_uploader("Upload credentials.json", type="json")
            
            if uploaded_file is not None:
                # Save uploaded credentials
                bytes_data = uploaded_file.getvalue()
                with open("credentials.json", "wb") as f:
                    f.write(bytes_data)
                st.success("Credentials file uploaded successfully!")
                
                if st.button("Authenticate with Google", key="auth_google_btn_credentials"):
                    with st.spinner("Authenticating..."):
                        try:
                            # Initialize agent
                            st.session_state.agent = GmailSpamAgent()
                            
                            # Authenticate
                            if st.session_state.agent.authenticate():
                                st.session_state.authenticated = True
                                add_log("Successfully authenticated with Google")
                                st.experimental_rerun()
                            else:
                                st.error("Authentication failed. Please check your credentials.")
                                add_log("Authentication failed", "error")
                        except Exception as e:
                            st.error(f"Error during authentication: {str(e)}")
                            add_log(f"Authentication error: {str(e)}", "error")
        else:
            st.success("✅ Authenticated with Google")
            
            if st.button("Sign Out", key="sign_out_btn_main"):
                # Clear token and reset state
                if os.path.exists("token.pickle"):
                    os.remove("token.pickle")
                st.session_state.authenticated = False
                st.session_state.agent = None
                add_log("Signed out successfully")
                st.experimental_rerun()
        
        # Settings section (only shown when authenticated)
        if st.session_state.authenticated:
            st.subheader("Settings")

            # MCP Integration
            st.checkbox(
                "Enable MCP Server Integration",
                value=False,
                key="mcp_integration_enabled",
                help="Use the MCP server for spam detection and deletion"
            )

            # Agent rules configuration
            with st.expander("Detection Settings", expanded=False):
                # Rule configurations
                st.slider(
                    "Max Spam Score",
                    min_value=0.5,
                    max_value=1.0,
                    value=st.session_state.agent.rules["max_spam_score"],
                    step=0.05,
                    key="max_spam_score",
                    help="Threshold for ML classifier to flag as spam"
                )
                
                st.slider(
                    "Min Confidence to Delete", 
                    min_value=0.5, 
                    max_value=1.0, 
                    value=st.session_state.agent.rules["min_confidence"],
                    step=0.05,
                    key="min_confidence",
                    help="Minimum confidence level required to auto-delete"
                )
                
                st.slider(
                    "Max Emails Per Run", 
                    min_value=10, 
                    max_value=500, 
                    value=st.session_state.agent.rules["max_emails_per_run"],
                    step=10,
                    key="max_emails_per_run",
                    help="Maximum number of emails to process in one scan"
                )
                
                st.slider(
                    "Email Age Threshold (days)", 
                    min_value=1, 
                    max_value=90, 
                    value=st.session_state.agent.rules["age_threshold_days"],
                    step=1,
                    key="age_threshold_days",
                    help="Don't process emails older than this"
                )
                
                st.checkbox(
                    "Ignore Contacts", 
                    value=st.session_state.agent.rules["ignore_contacts"],
                    key="ignore_contacts",
                    help="Skip emails from your contacts"
                )
                
                # Update agent rules from UI inputs
                if st.button("Save Settings", key="save_settings_btn"):
                    st.session_state.agent.rules["max_spam_score"] = st.session_state.max_spam_score
                    st.session_state.agent.rules["min_confidence"] = st.session_state.min_confidence
                    st.session_state.agent.rules["max_emails_per_run"] = st.session_state.max_emails_per_run
                    st.session_state.agent.rules["age_threshold_days"] = st.session_state.age_threshold_days
                    st.session_state.agent.rules["ignore_contacts"] = st.session_state.ignore_contacts
                    
                    add_log("Settings updated successfully")
                    st.success("Settings saved!")
            
            # Keyword management
            with st.expander("Spam Keywords", expanded=False):
                keywords_text = st.text_area(
                    "Enter spam keywords (one per line)",
                    value="\n".join(st.session_state.agent.rules["obvious_spam_keywords"]),
                    height=200
                )
                
                if st.button("Update Keywords", key="update_keywords_btn"):
                    new_keywords = [kw.strip() for kw in keywords_text.split("\n") if kw.strip()]
                    st.session_state.agent.rules["obvious_spam_keywords"] = new_keywords
                    add_log(f"Updated spam keywords list: {len(new_keywords)} keywords")
                    st.success("Keywords updated!")
            
            # Domain management
            with st.expander("Suspicious Domains", expanded=False):
                domains_text = st.text_area(
                    "Enter suspicious domains (one per line)",
                    value="\n".join(st.session_state.agent.rules["suspicious_email_domains"]),
                    height=200
                )
                
                if st.button("Update Domains", key="update_domains_btn"):
                    new_domains = [d.strip() for d in domains_text.split("\n") if d.strip()]
                    st.session_state.agent.rules["suspicious_email_domains"] = new_domains
                    add_log(f"Updated suspicious domains list: {len(new_domains)} domains")
                    st.success("Domains updated!")
            
            # Actions section
            st.subheader("Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Scan Inbox", key="scan_inbox_btn"):
                    with st.spinner("Scanning inbox for spam..."):
                        try:
                            # Train classifier if needed
                            if not st.session_state.agent.classifier:
                                add_log("Training spam classifier...")
                                st.session_state.agent.train_classifier()
                            # Scan inbox
                            if st.session_state.mcp_integration_enabled:
                                add_log("MCP server integration is enabled, but not yet implemented.", "warning")
                                st.session_state.spam_emails = []  # Placeholder
                                st.session_state.scan_results = {
                                    'timestamp': datetime.now(),
                                    'spam_detected': 0,
                                    'scanned_count': st.session_state.agent.rules["max_emails_per_run"]
                                }
                            else:
                                st.session_state.spam_emails = st.session_state.agent.scan_inbox()
                                st.session_state.scan_results = {
                                    'timestamp': datetime.now(),
                                    'spam_detected': len(st.session_state.spam_emails),
                                    'scanned_count': st.session_state.agent.rules["max_emails_per_run"]
                                }
                            # Add to history
                            if st.session_state.scan_results["spam_detected"] > 0:
                                st.session_state.history.append({
                                    'timestamp': datetime.now(),
                                    'action': 'scan',
                                    'spam_detected': st.session_state.scan_results["spam_detected"],
                                    'deleted': 0
                                })
                            add_log(f"Scan completed. Found {len(st.session_state.spam_emails)} potential spam emails.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error during scan: {str(e)}")
                            add_log(f"Scan error: {str(e)}", "error")
                # Add a date input for custom archive cutoff date
                archive_cutoff = st.date_input(
                    "Archive unread emails before this date",
                    value=datetime.now().replace(day=1),
                    key="archive_cutoff_date"
                )
                # Add the new button for archiving unread emails before the selected date
                if st.button("Archive Unread Emails Before Selected Date", key="archive_unread_btn"):
                    with st.spinner("Archiving unread emails before selected date..."):
                        try:
                            cutoff_str = archive_cutoff.strftime('%Y/%m/%d')
                            archived_count = st.session_state.agent.archive_unread_before_date(cutoff_str)
                            add_log(f"Archived {archived_count} unread emails before {cutoff_str}.")
                            st.success(f"Archived {archived_count} unread emails before {cutoff_str}.")
                        except Exception as e:
                            st.error(f"Error archiving unread emails: {str(e)}")
                            add_log(f"Error archiving unread emails: {str(e)}", "error")
            
            with col2:
                delete_btn = st.button(
                    "Delete Detected Spam",
                    disabled=(st.session_state.scan_results is None or len(st.session_state.spam_emails) == 0),
                    key="delete_spam_btn"
                )
                
                if delete_btn:
                    with st.spinner("Deleting spam emails..."):
                        try:
                            if st.session_state.mcp_integration_enabled:
                                add_log("MCP server integration is enabled, but not yet implemented.", "warning")
                                deleted_count = 0  # Placeholder
                            else:
                                deleted_count = st.session_state.agent.delete_spam(st.session_state.spam_emails)
                            
                            # Add to history
                            st.session_state.history.append({
                                'timestamp': datetime.now(),
                                'action': 'delete',
                                'spam_detected': len(st.session_state.spam_emails),
                                'deleted': deleted_count
                            })
                            
                            add_log(f"Deleted {deleted_count} spam emails")
                            st.session_state.spam_emails = []
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error deleting emails: {str(e)}")
                            add_log(f"Delete error: {str(e)}", "error")
            
            # Run in full mode (scan + delete)
            if st.button("Scan and Delete (Full Run)", key="full_run_btn"):
                with st.spinner("Running full spam management cycle..."):
                    try:
                        results = st.session_state.agent.run(learning_mode=False, delete=True)
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'action': 'full_run',
                            'spam_detected': results['spam_detected'],
                            'deleted': results['spam_deleted']
                        })
                        
                        add_log(f"Full run completed. Detected {results['spam_detected']} spam emails and deleted {results['spam_deleted']}.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error during full run: {str(e)}")
                        add_log(f"Full run error: {str(e)}", "error")
        
        # Display log
        st.subheader("Activity Log")
        if st.session_state.log_messages:
            log_df = pd.DataFrame(st.session_state.log_messages)
            for i, row in log_df.iloc[::-1].iterrows():
                if row["level"] == "error":
                    st.error(f"{row['timestamp']} - {row['message']}")
                elif row["level"] == "warning":
                    st.warning(f"{row['timestamp']} - {row['message']}")
                else:
                    st.info(f"{row['timestamp']} - {row['message']}")
        else:
            st.text("No activity logged yet")
        
        # Clear log button
        if st.button("Clear Log", key="clear_log_btn"):
            st.session_state.log_messages = []
            st.experimental_rerun()

        # Sidebar for authentication
        st.title("API Configuration")
        
        # OpenAI API Key Input
        st.subheader("OpenAI API Key")
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.openai_api_key or "")
        if openai_api_key and openai_api_key != st.session_state.openai_api_key:
            try:
                import openai as openai_module
                openai_module.api_key = openai_api_key
                openai_module.models.list()
                st.session_state.openai_api_key = openai_api_key
                st.success("OpenAI API Key validated successfully!")
            except Exception as e:
                st.error(f"Invalid OpenAI API Key: {str(e)}")
                st.session_state.openai_api_key = None
                return

        # Gmail Authentication
        st.subheader("Gmail Authentication")
        if not st.session_state.authenticated:
            if st.button("Authenticate with Gmail", key="auth_google_btn_sidebar"):
                if not st.session_state.openai_api_key:
                    st.error("Please enter a valid OpenAI API Key first")
                    return
                st.session_state.authenticated = True
                st.session_state.agent = GmailSpamAgent()
                st.success("Successfully authenticated!")
        else:
            st.success("Authenticated with Gmail")
            if st.button("Sign Out", key="sign_out_btn_sidebar"):
                st.session_state.authenticated = False
                st.session_state.agent = None
                st.rerun()

        # --- Unified LangChain Agent Interface ---
        st.header("🤖 Gmail Assistant")
        st.markdown("""
        Ask me to help with your Gmail tasks. I can:
        - Read emails from any label
        - Delete emails
        - Move emails to different labels
        - Show unread messages
        - And more!
        
        Examples:
        - "Show me unread emails from last week"
        - "Delete all spam emails"
        - "Move unread emails to Finance label"
        - "Read my latest emails from INBOX"
        """)
        
        # Initialize LangChain agent if we have the API key
        if st.session_state.openai_api_key:
            llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=st.session_state.openai_api_key)
            
            # Define the tools with proper function signatures using @tool decorator and type hints
            @tool
            def read_emails_tool(label: str = "INBOX", unread_only: bool = False, max_to_fetch: int = None) -> list:
                """Read emails from a specified label.
                Args:
                    label: The Gmail label to read from (e.g., "INBOX", "SPAM")
                    unread_only: Set to True to only get unread emails
                    max_to_fetch: Maximum number of emails to fetch (None for unlimited)
                Returns:
                    List of email objects that can be used by other tools
                """
                if not st.session_state.authenticated:
                    return "Please authenticate with Gmail first"
                try:
                    if label.lower() == "inbox":
                        if unread_only:
                            return st.session_state.agent.fetch_all_unread_emails(max_to_fetch=max_to_fetch)
                        else:
                            return st.session_state.agent.fetch_emails_after_date("2000/01/01", max_to_fetch=max_to_fetch)
                    elif label.lower() == "spam":
                        if unread_only:
                            return st.session_state.agent.fetch_unread_spam_emails(max_to_fetch=max_to_fetch or 100)
                        else:
                            return st.session_state.agent.scan_inbox()
                    else:
                        return st.session_state.agent.fetch_emails_after_date("2000/01/01", max_to_fetch=max_to_fetch, label=label)
                except Exception as e:
                    return f"Error fetching emails: {str(e)}"

            @tool
            def move_emails_tool(email_ids: list, label: str) -> str:
                """Move emails to a specified label.
                Args:
                    email_ids: List of email IDs to move
                    label: The label to move the emails to
                Returns:
                    Status message about the move operation
                """
                if not st.session_state.authenticated:
                    return "Please authenticate with Gmail first"
                try:
                    if not email_ids:
                        return "No email IDs provided"
                    emails = [{'id': eid} for eid in email_ids]
                    moved = st.session_state.agent.move_emails_to_label(emails, label)
                    return f"Moved {moved} emails to label '{label}'."
                except Exception as e:
                    return f"Error moving emails: {str(e)}"

            @tool
            def delete_emails_tool(email_ids: list) -> str:
                """Delete emails by their IDs.
                Args:
                    email_ids: List of email IDs to delete
                Returns:
                    Status message about the delete operation
                """
                return delete_emails(email_ids=email_ids)

            tools = [read_emails_tool, move_emails_tool, delete_emails_tool]
            
            # Initialize the agent with a specific system message
            langchain_agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                handle_parsing_errors=True,
                system_message="""You are a Gmail assistant that helps users manage their emails.
                When asked to move or delete emails:
                1. First use read_emails to fetch the emails
                2. Then use move_emails or delete_emails with the fetched email IDs
                3. Always perform both steps - don't just fetch without taking action
                4. If moving emails, create the label if it doesn't exist
                5. Report the total number of emails processed
                
                For read_emails:
                - Use label="INBOX" for inbox emails
                - Set unread_only=True for unread emails
                - Use max_to_fetch to limit the number of emails (e.g., 200)
                """
            )
            
            # Always show the prompt input, but disable if not authenticated
            prompt = st.text_input(
                "What would you like me to do with your emails?",
                key="gmail_assistant_prompt",
                disabled=not st.session_state.authenticated
            )
            
            if st.button("Send", key="gmail_assistant_send", disabled=not st.session_state.authenticated):
                if not st.session_state.authenticated:
                    st.warning("Please authenticate with Gmail first")
                else:
                    with st.spinner("Processing your request..."):
                        try:
                            # Add context to the prompt to ensure action is taken
                            enhanced_prompt = f"""
                            User request: {prompt}
                            
                            Remember to:
                            1. First fetch the relevant emails using read_emails with appropriate parameters
                            2. Then take the requested action (move/delete) using the fetched email IDs
                            3. Report the total number of emails processed
                            """
                            result = langchain_agent.run(enhanced_prompt)
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"Error processing request: {str(e)}")
        else:
            st.warning("Please enter your OpenAI API key to use the Gmail Assistant")

def display_dashboard():
    """Display the main dashboard with statistics and email list."""
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Spam Emails", "History"])
    
    # Dashboard tab
    with tab1:
        # Display stats in cards
        if st.session_state.authenticated:
            st.subheader("Spam Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            # Card 1: Last scan time
            with col1:
                st.markdown("### Last Scan")
                if st.session_state.scan_results:
                    st.markdown(f"**Time:** {st.session_state.scan_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**Spam Detected:** {st.session_state.scan_results['spam_detected']}")
                    st.markdown(f"**Emails Scanned:** {st.session_state.scan_results['scanned_count']}")
                else:
                    st.markdown("No scans performed yet")
            
            # Card 2: Total stats
            with col2:
                st.markdown("### Total Stats")
                if st.session_state.history:
                    total_detected = sum(h['spam_detected'] for h in st.session_state.history)
                    total_deleted = sum(h['deleted'] for h in st.session_state.history)
                    st.markdown(f"**Total Spam Detected:** {total_detected}")
                    st.markdown(f"**Total Spam Deleted:** {total_deleted}")
                    st.markdown(f"**Actions Performed:** {len(st.session_state.history)}")
                else:
                    st.markdown("No stats available yet")
            
            # Card 3: Current settings
            with col3:
                st.markdown("### Current Settings")
                if st.session_state.agent:
                    st.markdown(f"**Confidence Threshold:** {st.session_state.agent.rules['min_confidence']}")
                    st.markdown(f"**Max Emails Per Run:** {st.session_state.agent.rules['max_emails_per_run']}")
                    st.markdown(f"**Spam Keywords:** {len(st.session_state.agent.rules['obvious_spam_keywords'])}")
                else:
                    st.markdown("Agent not initialized")
            
            # Charts row
            st.subheader("Analytics")
            if st.session_state.history:
                col1, col2 = st.columns(2)
                
                # Historical chart
                with col1:
                    st.markdown("### Detection History")
                    
                    # Create history dataframe
                    history_df = pd.DataFrame(st.session_state.history)
                    history_df['date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history_df['date'], history_df['spam_detected'], 'b-o', label='Detected')
                    ax.plot(history_df['date'], history_df['deleted'], 'r-o', label='Deleted')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Count')
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Pie chart
                with col2:
                    st.markdown("### Spam Management Stats")
                    
                    total_detected = sum(h['spam_detected'] for h in st.session_state.history)
                    total_deleted = sum(h['deleted'] for h in st.session_state.history)
                    
                    if total_detected > 0:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(
                            [total_deleted, total_detected - total_deleted],
                            labels=['Deleted', 'Not Deleted'],
                            autopct='%1.1f%%',
                            colors=['#ff9999','#66b3ff']
                        )
                        ax.axis('equal')
                        st.pyplot(fig)
                    else:
                        st.info("No spam detected yet")
            else:
                st.info("Run a scan to see analytics")
        else:
            st.info("Please authenticate with your Google account in the sidebar to view the dashboard.")

    # Spam emails tab
    with tab2:
        if st.session_state.authenticated:
            st.subheader("Detected Spam Emails")
            
            if st.session_state.spam_emails:
                for i, spam in enumerate(st.session_state.spam_emails):
                    with st.expander(f"{i+1}. {st.session_state.agent._get_subject(spam['message'])} - {st.session_state.agent._get_sender(spam['message'])}"):
                        # Email details
                        st.markdown(f"**From:** {st.session_state.agent._get_sender(spam['message'])}")
                        st.markdown(f"**Subject:** {st.session_state.agent._get_subject(spam['message'])}")
                        st.markdown(f"**Confidence:** {spam['confidence']:.2f}")
                        
                        # Reasons
                        st.markdown("**Reasons flagged as spam:**")
                        for reason in spam['reasons']:
                            st.markdown(f"- {reason}")
                        
                        # Email body preview
                        st.markdown("**Email Preview:**")
                        body_text = st.session_state.agent._extract_email_text(spam['message']) or "No text content available"
                        st.text_area("", body_text, height=150, disabled=True)
                        
                        # Action buttons for individual email
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Delete Email #{i+1}", key=f"delete_email_{i}"):
                                try:
                                    st.session_state.agent.delete_spam([spam])
                                    st.session_state.spam_emails.pop(i)
                                    add_log(f"Deleted email: {st.session_state.agent._get_subject(spam['message'])}")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error deleting email: {str(e)}")
                                    add_log(f"Delete error: {str(e)}", "error")
                        
                        with col2:
                            if st.button(f"Not Spam #{i+1}", key=f"not_spam_{i}"):
                                st.session_state.spam_emails.pop(i)
                                add_log(f"Marked as not spam: {st.session_state.agent._get_subject(spam['message'])}")
                                st.experimental_rerun()
            else:
                st.info("No spam emails detected in the latest scan.")
        else:
            st.info("Please authenticate with your Google account in the sidebar to view spam emails.")

    # History tab
    with tab3:
        if st.session_state.authenticated:
            st.subheader("Action History")
            
            if st.session_state.history:
                history_df = pd.DataFrame(st.session_state.history)
                history_df['time'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Format the dataframe for display
                display_df = history_df[['time', 'action', 'spam_detected', 'deleted']].copy()
                display_df.columns = ['Time', 'Action', 'Spam Detected', 'Deleted']
                display_df['Action'] = display_df['Action'].map({
                    'scan': 'Scan Only', 
                    'delete': 'Delete',
                    'full_run': 'Full Scan & Delete'
                })
                
                st.dataframe(display_df, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export History to CSV", key="export_history_btn"):
                        csv = display_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="spam_history.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Clear History", key="clear_history_btn"):
                        st.session_state.history = []
                        add_log("History cleared")
                        st.experimental_rerun()
            else:
                st.info("No history available yet.")
        else:
            st.info("Please authenticate with your Google account in the sidebar to view history.")

def display_welcome():
    """Display welcome message for new users, including all latest features."""
    st.markdown("""
    ## Welcome to Gmail Spam Management Agent!
    
    This application helps you manage your Gmail inbox using advanced AI and machine learning techniques:
    
    ### ✨ Core Features
    1. **AI-Powered Spam Detection**
       - Machine learning classifier for accurate spam detection
       - Rule-based filtering with customizable keywords
       - Suspicious domain detection
       - Confidence scoring for each detection
    
    2. **Smart Email Management**
       - Bulk spam detection and deletion
       - Custom archive creation for unread emails
       - Email age-based filtering
       - Contact-based filtering
    
    3. **Advanced Configuration**
       - Adjustable spam detection thresholds
       - Customizable confidence levels
       - Configurable email processing limits
       - Age-based filtering settings
    
    4. **Interactive Dashboard**
       - Real-time spam statistics
       - Visual analytics and charts
       - Action history tracking
       - Exportable reports
    
    5. **AI Assistant Integration**
       - Natural language email management
       - Smart email organization
       - Automated email handling
       - Custom label management
    
    6. **Security & Privacy**
       - Local authentication handling
       - Secure credential management
       - No data storage on external servers
       - OAuth 2.0 security
    
    ---
    
    ## Getting Started
    1. **Authentication**: Use the sidebar to authenticate with your Google account
    2. **Configuration**: Set up your spam detection preferences
    3. **Scan**: Run a scan to identify potential spam
    4. **Review**: Check the identified spam emails
    5. **Manage**: Delete or archive unwanted emails
    6. **Monitor**: Track your spam management history
    
    ---
    
    ### Security Note
    This application runs entirely on your computer. Your authentication credentials are stored locally and are only used to access your Gmail account securely.
    """, unsafe_allow_html=True)
    
    # Quick start button
    if st.button("Get Started"):
        st.session_state.first_time = False
        st.experimental_rerun()

def gemini_chat(query, api_key, context=None):
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model with basic configuration
        model = genai.GenerativeModel('gemini-pro')
        
        # Add context if provided
        if context:
            full_query = f"Context: {context}\n\nQuery: {query}"
        else:
            full_query = query
            
        # Generate response
        response = model.generate_content(full_query)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return "Error: Invalid model name or API version. Please check your API key and try again."
        elif "401" in error_msg:
            return "Error: Invalid API key. Please check your API key and try again."
        else:
            return f"Error from Gemini API: {error_msg}"

def hf_chat(prompt, api_key, model="meta-llama/Meta-Llama-3-8B"):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "error" in result:
            return f"Error: {result['error']}"
        else:
            return str(result)
    except Exception as e:
        return f"Error from Hugging Face API: {e}"

def groq_chat(prompt, api_key, model="llama3-8b-8192", context=None):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Inject context if provided
    if context:
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}"
    else:
        full_prompt = prompt
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error from Groq API: {e}"

def main():
    """Main application function."""
    # Initialize session state variables FIRST
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None

    # Check if first time
    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
    
    # Set up the UI
    display_header()
    setup_sidebar()
    
    # Main content
    if st.session_state.first_time:
        display_welcome()
    else:
        display_welcome()

if __name__ == "__main__":
    main()
