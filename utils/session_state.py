import streamlit as st

def initialize_session_state():
    """
    Initialize all session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
        
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "General chat"
        
    if "response_style" not in st.session_state:
        st.session_state.response_style = "Detailed"

def clear_chat_history():
    """
    Clear the chat messages in session state.
    """
    st.session_state.messages = []

def add_message(role, content):
    """
    Add a message to the chat history.
    """
    st.session_state.messages.append({"role": role, "content": content})
