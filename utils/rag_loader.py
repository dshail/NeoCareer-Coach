import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_docs(file_paths, chunk_size=1000, chunk_overlap=100):
    """
    Load documents from file paths and split them into chunks.
    """
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue # Skip unsupported formats
            
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)
