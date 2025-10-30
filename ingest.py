
import os
import logging
from src import pipeline
from add_books import add_new_islamic_books # <-- Import your smart function

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_medical_data():
    """
    Handles the ingestion for both medical CSV and document files.
    NOTE: This is a simple version that re-processes all files.
    For a fully robust system, you would build incremental logic
    similar to 'add_books.py' for these sources as well.
    """
    logger.info("--- Processing Medical Data ---")

    # Ingest Medical CSV (assuming this needs to be run each time)
    logger.info("Processing medical data from 'medquad.csv'...")
    try:
        pipeline.pipeline(
            inputPath="medquad.csv",
            parser_name=None,
            chunking_strategy=None,
            retrieval_strategy="agentic",
            input_type="csv"
        )
        logger.info("✅ Medical CSV processed.")
    except Exception as e:
        logger.error(f"❌ Failed to ingest medical CSV: {e}")

    # Ingest Medical Documents
    logger.info("\nProcessing medical documents from 'docs' folder...")
    try:
        pipeline.pipeline(
            inputPath="docs",
            parser_name="pymupdf4llm",
            chunking_strategy="semantic",
            retrieval_strategy="agentic",
            cli=False
        )
        logger.info("✅ Medical documents processed.")
    except Exception as e:
        logger.error(f"❌ Failed to ingest medical documents: {e}")

def ingest_islamic_data():
    """
    Handles the incremental addition of new Islamic texts.
    """
    logger.info("\n--- Processing Islamic Texts (Incremental Update) ---")
    if not os.path.exists("islamic_texts"):
        logger.warning("📁 'islamic_texts' folder not found, skipping.")
        return

    # Metadata for your Islamic books (can be kept here or in add_books.py)
    book_metadata = {
        "umdat_al-salik.pdf": { "madhab": "shafii", "type": "fiqh", "author": "Ahmad ibn Naqib al-Misri", "title": "Umdat al-Salik" },
        "minhaj_at_talibin.pdf": { "madhab": "shafii", "type": "fiqh", "author": "Imam Nawawi", "title": "Minhaj al-Talibin" },
        "sahih_bukhari.pdf": { "madhab": "general", "type": "hadith", "collection": "sahih_bukhari" }
    }

    add_new_islamic_books(
        folder_path="islamic_texts",
        book_metadata=book_metadata
    )

if __name__ == "__main__":
    logger.info("======================================================")
    logger.info("🚀 STARTING UNIFIED DATA INGESTION AND UPDATE SCRIPT 🚀")
    logger.info("======================================================")

    # You can run all ingestions, or comment out the ones you don't need
    
    # Run medical data ingestion
    #ingest_medical_data()
    
    # Run Islamic data ingestion (will only add new books)
    ingest_islamic_data()

    logger.info("======================================================")
    logger.info("✅ All ingestion tasks complete.")
    logger.info("======================================================")