import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# --- NEW: Import PyMuPDFLoader ---
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

# --- 1. SETUP ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# --- 2. CONFIGURATION ---
DATA_PATHS = ["CarTakaful", "MotorTakaful"]  # Folders containing your PDFs
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "etiqa_Agentic_retrieval"
EMBEDDING_MODEL = "models/text-embedding-004"

# Chunking parameters
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

def load_documents_from_folders(paths):
    """
    Loads all PDF files from a list of specified folders using PyMuPDFLoader.
    """
    all_docs = []
    for path in paths:
        if not os.path.isdir(path):
            logger.warning(f"Directory not found, skipping: {path}")
            continue
        
        logger.info(f"--- Loading documents from: {path} ---")
        for filename in os.listdir(path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(path, filename)
                logger.info(f"Loading PDF: {file_path}")
                try:
                    # --- THE FIX ---
                    # Replaced UnstructuredPDFLoader with PyMuPDFLoader.
                    # This is a pure-Python library and does not need poppler.
                    loader = PyMuPDFLoader(file_path)
                    # --- END OF FIX ---
                    
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
                    
    return all_docs

def create_vector_store():
    """
    Main function to orchestrate the ingestion process.
    """
    logger.info("Starting ingestion process for Etiqa Takaful documents...")
    
    # 1. Load documents
    try:
        documents = load_documents_from_folders(DATA_PATHS)
        if not documents:
            logger.error("No documents were loaded. Please check your DATA_PATHS and PDF files.")
            return
        logger.info(f"Successfully loaded {len(documents)} document pages.")
    except Exception as e:
        logger.error(f"An error occurred during document loading: {e}", exc_info=True)
        return

    # 2. Split documents into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            logger.error("Document splitting resulted in 0 chunks.")
            return
        logger.info(f"Split documents into {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"An error occurred during document splitting: {e}", exc_info=True)
        return

    # 3. Initialize embedding model
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
        logger.info(f"Embedding model '{EMBEDDING_MODEL}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return

    # 4. Create and persist the vector database
    try:
        logger.info(f"Creating new vector store in '{CHROMA_DB_PATH}' with collection '{COLLECTION_NAME}'...")
        # This will create or overwrite the collection in the persistent directory
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH
        )
        logger.info(f"✅ Successfully created and persisted vector store with {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}", exc_info=True)

if __name__ == "__main__":
    create_vector_store()