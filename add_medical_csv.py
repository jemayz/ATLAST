

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document

# Assuming 'indexing' is still in src.doc_qa for now
from src.indexing import indexing 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def add_new_medical_csv(csv_file_path: str, collection_name: str = "medical_csv_Agentic_retrieval"):
    """
    Parses a CSV file with 'question' and 'answer' columns and adds it to the specified ChromaDB collection.
    """
    logger.info("=" * 60)
    logger.info(f"💊 MEDICAL CSV ADDITION SCRIPT")
    logger.info(f"🎯 Target Collection: {collection_name}")
    logger.info("=" * 60)
    
    if not os.path.exists(csv_file_path):
        logger.error(f"❌ Error: CSV file not found at path: {csv_file_path}")
        return False

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logger.error(f"❌ Error reading CSV file: {e}")
        return False
        
    required_columns = {'question', 'answer'}
    if not required_columns.issubset(df.columns):
        logger.error(f"❌ Error: CSV must contain columns: {required_columns}")
        logger.info(f"Found columns: {list(df.columns)}")
        return False
        
    logger.info(f"📊 Processing {len(df)} rows from {csv_file_path}...")
    
    new_chunks = []
    
    for index, row in df.iterrows():
        try:
            # 1. Use the 'answer' as the page content (the core knowledge)
            page_content = row['answer']
            
            # 2. Create rich metadata
            metadata = {
                'question': row['question'], # The question for retrieval augmentation
                'source': os.path.basename(csv_file_path), # File name as source
                'focus_area': 'medical', 
                'condition': 'general_medical', # General category for this CSV
                'domain': 'medical' 
            }
            
            # 3. Create the LangChain Document
            chunk = Document(page_content=page_content, metadata=metadata)
            new_chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"❌ Failed to process row {index}: {str(e)}")
            continue

    if not new_chunks:
        logger.warning("⚠️ No valid data rows were processed.")
        return False

    logger.info(f"📦 Adding {len(new_chunks)} total chunks to collection...")
    
    indexer = indexing()
    
    # Use the add_new_documents method which is designed for this exact purpose
    success = indexer.add_new_documents(
        documents=new_chunks,
        collection_name=collection_name
    )
    
    if success:
        logger.info(f"🎉 Successfully added {len(new_chunks)} new document chunks to {collection_name}!")
        return True
    else:
        logger.error("❌ Failed to add documents to collection.")
        return False

if __name__ == "__main__":
    CSV_PATH = "medical_dataset/medqa.csv" 
    
    add_new_medical_csv(CSV_PATH)
