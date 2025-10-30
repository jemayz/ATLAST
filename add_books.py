# add_books.py
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.indexing import indexing
from src.pipeline import DocParser, Chunker

def get_processed_books(collection_name="islamic_texts_Agentic_retrieval"):
    """Get list of books already in the collection"""
    try:
        indexer = indexing()
        collection = indexer.chroma_client.get_collection(name=collection_name)
        
        results = collection.get(include=['metadatas'])
        
        processed_books = set()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    book_name = Path(metadata['source']).name
                    processed_books.add(book_name)
        
        return processed_books
    except Exception as e:
        logger.warning(f"Could not get processed books: {e}")
        return set()

def add_new_islamic_books(folder_path="islamic_texts", 
                          collection_name="islamic_texts_Agentic_retrieval",
                          book_metadata=None):
    """Add new Islamic books to existing collection"""
    
    logger.info(f"🔍 Checking for new books in {folder_path}...")
    
    processed_books = get_processed_books(collection_name)
    logger.info(f"📚 Already processed: {processed_books}")
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    new_books = [f for f in pdf_files if f not in processed_books]
    
    if not new_books:
        logger.info("✅ No new books to process - all books already in collection")
        return True
    
    logger.info(f"📖 Found {len(new_books)} new book(s) to process: {new_books}")
    logger.info("⏰ Starting processing...")
    
    indexer = indexing()
    parser = DocParser("pymupdf4llm")
    chunker = Chunker("semantic")
    
    all_new_chunks = []
    
    for book_file in new_books:
        file_path = os.path.join(folder_path, book_file)
        
        try:
            logger.info(f"\n📄 Processing: {book_file}")
            logger.info(f"   📖 Parsing...")
            
            text_docs = parser.parse(file_path)
            
            if not text_docs:
                logger.warning(f"   ❌ {book_file} returned no content")
                continue
            
            logger.info(f"   ✅ Parsed {len(text_docs)} pages")
            logger.info(f"   ✂️ Chunking...")
            
            book_chunks = chunker.build_chunks(text_docs, source=file_path)
            
            if not book_chunks:
                logger.warning(f"   ❌ {book_file} returned no chunks")
                continue
            
            logger.info(f"   ✅ Created {len(book_chunks)} chunks")
            
            for chunk in book_chunks:
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                
                chunk.metadata['domain'] = 'islamic_texts'
                chunk.metadata['book_file'] = book_file
                
                if book_metadata and book_file in book_metadata:
                    chunk.metadata.update(book_metadata[book_file])
            
            all_new_chunks.extend(book_chunks)
            logger.info(f"   ✅ {book_file} processed successfully!")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process {book_file}: {str(e)}")
            continue
    
    if all_new_chunks:
        logger.info(f"\n📦 Adding {len(all_new_chunks)} total chunks to collection...")
        
        success = indexer.add_new_documents(
            documents=all_new_chunks,
            collection_name=collection_name
        )
        
        if success:
            logger.info(f"🎉 Successfully added {len(new_books)} new book(s)!")
            logger.info(f"📊 Books added: {new_books}")
            return True
        else:
            logger.error("❌ Failed to add documents to collection")
            return False
    else:
        logger.warning("⚠️ No new chunks to add")
        return False

if __name__ == "__main__":
    # Define metadata for your Islamic books
    book_metadata = {
        "umdat_al-salik.pdf": {
            "madhab": "shafii",
            "type": "fiqh",
            "author": "Ahmad ibn Naqib al-Misri",
            "title": "Umdat al-Salik"
        },
        "minhaj_at_talibin.pdf": {
            "madhab": "shafii", 
            "type": "fiqh",
            "author": "Imam Nawawi",
            "title": "Minhaj al-Talibin"
        },
        "sahih_bukhari.pdf": {
            "madhab": "general",
            "type": "hadith",
            "collection": "sahih_bukhari",
        }
    }
    
    logger.info("=" * 60)
    logger.info("🕌 ISLAMIC BOOKS ADDITION SCRIPT")
    logger.info("=" * 60)
    
    add_new_islamic_books(
        folder_path="islamic_texts",
        book_metadata=book_metadata
    )
    
    logger.info("=" * 60)
    logger.info("✅ Script completed!")
    logger.info("=" * 60)
    # ------------------ TEST CONFIGURATION ------------------
    # TEST_FOLDER = "islamic_texts_test" 
    # TEST_COLLECTION = "islamic_agentic_test_collection"
    
    # # Define metadata for the books you are TESTING
    # book_metadata = {
    #     # Only include the metadata for the book you put in the test folder
    #     "umdat_al-salik2.pdf": {
    #         "madhab": "shafii",
    #         "type": "fiqh",
    #         "author": "Ahmad ibn Naqib al-Misri",
    #         "title": "Umdat al-Salik"
    #     }
    #     # You MUST also include any books you may have in the main folder 
    #     # if the script is run with the main folder path later, 
    #     # but for this TEST, only the test book is needed.
    # }
    # # ---------------------------------------------------------
    
    # logger.info("=" * 60)
    # logger.info("🕌 ISLAMIC BOOKS ADDITION SCRIPT - AGENTIC CHUNKER TEST")
    # logger.info("=" * 60)
    
    # # *** ADD TIMER FOR MEASUREMENT ***
    # import time
    # start_time = time.time()
    
    # add_new_islamic_books(
    #     folder_path=TEST_FOLDER,
    #     collection_name=TEST_COLLECTION, # <-- Targets the new collection
    #     book_metadata=book_metadata
    # )
    
    # end_time = time.time()
    
    # logger.info("=" * 60)
    # logger.info(f"✅ Script completed in {end_time - start_time:.2f} seconds!")
    # logger.info("=" * 60)