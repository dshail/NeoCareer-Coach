import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY

def get_chat_model(provider="Groq", model_name=None, temperature=0.7):
    """
    Initialize and return a chat model based on the provider.
    """
    try:
        if provider == "OpenAI":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API Key not found.")
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=model_name or "gpt-4o",
                temperature=temperature
            )
        
        elif provider == "Groq":
            if not GROQ_API_KEY:
                raise ValueError("Groq API Key not found.")
            return ChatGroq(
                api_key=GROQ_API_KEY,
                model=model_name or "llama-3.3-70b-versatile",
                temperature=temperature
            )
        
        elif provider == "Gemini":
            if not GOOGLE_API_KEY:
                raise ValueError("Google API Key not found.")
            return ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY, # Fixed parameter name
                model=model_name or "gemini-1.5-pro",
                temperature=temperature
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {provider} model: {str(e)}")

# For backward compatibility with existing code
def get_chatgroq_model():
    return get_chat_model(provider="Groq")