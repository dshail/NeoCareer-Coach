import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_chat_model
from models.embeddings import get_embedding_model
from utils.rag_loader import load_and_chunk_docs
from utils.rag_retriever import create_vector_store, get_relevant_context
from utils.web_search import search_web
from utils.prompts import get_system_prompt
from utils.session_state import initialize_session_state, clear_chat_history, add_message
from config.config import DEFAULT_TEMPERATURE

def get_chat_response(chat_model, messages, system_prompt, context=None, web_results=None):
    """Get response from the chat model with optional context and search results"""
    try:
        full_system_prompt = system_prompt
        if context:
            full_system_prompt += f"\n\nRetrieved Document Context:\n{context}"
        if web_results:
            full_system_prompt += f"\n\nLive Web Search Results:\n{web_results}"
            
        formatted_messages = [SystemMessage(content=full_system_prompt)]
        
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def main():
    st.set_page_config(
        page_title="NeoCareer Coach - Your AI Prep Partner",
        page_icon="🚀",
        layout="wide"
    )

    initialize_session_state()

    # --- Sidebar ---
    with st.sidebar:
        st.title("🚀 NeoCareer Coach")
        st.markdown("---")
        
        # Provider & Model Settings
        st.header("⚙️ Settings")
        provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Gemini"], index=0)
        
        if provider == "Groq":
            model_name = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        elif provider == "OpenAI":
            model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        else:
            model_name = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"])
            
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE)
        
        # Domain Mode & Response Style
        st.divider()
        st.header("🎯 Mode")
        st.session_state.current_mode = st.radio(
            "Select Domain Mode:",
            ["General chat", "Gap analysis", "Mock interview", "Company research", "Prep planner"],
            index=0
        )
        
        st.session_state.response_style = st.toggle("Concise Mode", value=False)
        style_text = "Concise" if st.session_state.response_style else "Detailed"
        
        # Document Upload
        st.divider()
        st.header("📁 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload Resume, JD, or Notes (PDF/DOCX)", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files:
                with st.spinner("Processing documents..."):
                    temp_paths = []
                    for uploaded_file in new_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_paths.append(tmp.name)
                    
                    try:
                        chunks = load_and_chunk_docs(temp_paths)
                        # Invoke embedding model inside app.py
                        embeddings = get_embedding_model(provider="Google")
                        vector_store = create_vector_store(chunks, embeddings)
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_files.extend([f.name for f in new_files])
                            st.success(f"Processed {len(new_files)} new documents!")
                        else:
                            st.error("Failed to process documents.")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                    finally:
                        # Cleanup temp files
                        for p in temp_paths: 
                            if os.path.exists(p):
                                os.remove(p)

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()

    # --- Main Chat ---
    st.title(f"🤖 {st.session_state.current_mode} Mode")
    st.caption(f"Status: Playing as your **{st.session_state.current_mode}** | Style: **{style_text}**")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me anything about your career prep..."):
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing and fetching context..."):
                # Initialize variables
                context = None
                web_results = None
                
                # Hybrid Retrieval Logic
                if st.session_state.vector_store:
                    context = get_relevant_context(st.session_state.vector_store, prompt)
                
                # Use web search for company research or if explicitly requested/needed
                if st.session_state.current_mode == "Company research" or "latest" in prompt.lower() or "current" in prompt.lower():
                    web_results = search_web(prompt)

                # Get model
                try:
                    chat_model = get_chat_model(provider, model_name, temperature)
                    system_prompt = get_system_prompt(st.session_state.current_mode, style_text)
                    
                    response = get_chat_response(chat_model, st.session_state.messages, system_prompt, context, web_results)
                    st.markdown(response)
                    add_message("assistant", response)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()