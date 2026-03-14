from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model

def create_vector_store(chunks, embeddings):
    """
    Create an in-memory FAISS vector store from document chunks.
    """
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

def get_relevant_context(vector_store, query, k=5):
    """
    Retrieve relevant chunks from the vector store based on query.
    """
    if not vector_store:
        return ""
    
    try:
        docs = vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return ""
