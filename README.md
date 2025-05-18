# Gmail Spam Management Agent

An intelligent agent that uses AI and machine learning to detect and manage spam emails in your Gmail account.

## ✨ Features

### 1. AI-Powered Spam Detection
- Machine learning classifier for accurate spam detection
- Rule-based filtering with customizable keywords
- Suspicious domain detection
- Confidence scoring for each detection

### 2. Smart Email Management
- Bulk spam detection and deletion
- Custom archive creation for unread emails
- Email age-based filtering
- Contact-based filtering

### 3. Advanced Configuration
- Adjustable spam detection thresholds
- Customizable confidence levels
- Configurable email processing limits
- Age-based filtering settings

### 4. Interactive Dashboard
- Real-time spam statistics
- Visual analytics and charts
- Action history tracking
- Exportable reports

### 5. AI Assistant Integration
- Natural language email management
- Smart email organization
- Automated email handling
- Custom label management

### 6. Security & Privacy
- Local authentication handling
- Secure credential management
- No data storage on external servers
- OAuth 2.0 security

## Prerequisites

- Python 3.7 or higher
- Google account with Gmail
- Google Cloud Project with Gmail API enabled
- OpenAI API key (for AI assistant features)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gmail-spam-agent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Cloud Project:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable the Gmail API
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `credentials.json` in the project directory

4. Set up OpenAI API (for AI assistant features):
   - Get your API key from [OpenAI Platform](https://platform.openai.com)
   - Add it to the application through the UI

## Running the Application

1. Start the Streamlit interface:
   ```bash
   python -m streamlit run gmail_spam_agent_streamlit.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Follow the setup wizard:
   - Upload your `credentials.json` file
   - Authenticate with your Google account
   - Configure your spam detection settings
   - Add your OpenAI API key (optional, for AI assistant features)

## Configuration Guide

### Spam Detection Settings
- **Max Spam Score**: Threshold for ML classifier (0.5-1.0)
- **Min Confidence**: Required confidence for auto-deletion (0.5-1.0)
- **Max Emails Per Run**: Limit for batch processing (10-500)
- **Email Age Threshold**: Don't process emails older than X days (1-90)

### Custom Rules
- **Spam Keywords**: Add custom keywords to detect spam
- **Suspicious Domains**: List of domains to flag as suspicious
- **Ignore Contacts**: Skip emails from your contacts

### AI Assistant Features
- Natural language commands for email management
- Smart email organization
- Automated email handling
- Custom label management

## Security & Privacy

- All authentication is handled locally
- Credentials are stored securely using OAuth 2.0
- No email data is stored on external servers
- All processing happens on your local machine

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 