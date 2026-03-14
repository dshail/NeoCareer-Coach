import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key_name, default=""):
    """
    Retrieve secret from Streamlit secrets (for cloud deployment) 
    or from environment variables (for local deployment).
    """
    try:
        # Try Streamlit Secrets first (Cloud)
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        # Not running in a Streamlit context or secrets not configured
        pass
        
    # Fallback to environment variables (Local)
    return os.getenv(key_name, default)

# API Keys
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# Default Models
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"

# Embedding Models
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"

# App Settings
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS_CONCISE = 500
MAX_TOKENS_DETAILED = 2000
