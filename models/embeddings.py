from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.config import OPENAI_API_KEY, GOOGLE_API_KEY, OPENAI_EMBEDDING_MODEL, GOOGLE_EMBEDDING_MODEL

def get_embedding_model(provider="Google"):
    """
    Initialize and return an embedding model based on the provider.
    """
    try:
        if provider == "OpenAI":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API Key not found.")
            return OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model=OPENAI_EMBEDDING_MODEL
            )
        
        elif provider == "Google":
            if not GOOGLE_API_KEY:
                raise ValueError("Google API Key not found.")
            # Explicitly print model to debug if it persists
            print(f"DEBUG: Initializing Google Embeddings with model: {GOOGLE_EMBEDDING_MODEL}")
            return GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, # Use google_api_key instead of api_key for GoogleGenerativeAIEmbeddings
                model=GOOGLE_EMBEDDING_MODEL or "models/text-embedding-004"
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {provider} embeddings: {str(e)}")
