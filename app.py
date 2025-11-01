from flask import Flask, request, render_template, session, url_for,redirect
from flask_session import Session
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import json
import logging
import re
import traceback
from PIL import Image
import base64
from api import api_bp

# --- Core Application Imports ---
from src.medical_swarm import run_medical_swarm
from src.utils import load_rag_system,standardize_query,get_standalone_question,parse_agent_response,markdown_bold_to_html
from langchain_google_genai import ChatGoogleGenerativeAI


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global store for chat history
medical_chat_history = []
islamic_chat_history = []
insurance_chat_history = []

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session signing

# --- CONFIGURE SERVER-SIDE SESSIONS ---
app.config["SESSION_PERMANENT"]= False
app.config["SESSION_TYPE"]= "filesystem"
Session(app)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=google_api_key)

#START    
logger.info("🌟 Starting Multi-Domain AI Assistant...")

rag_systems = {
    'medical': load_rag_system(collection_name="medical_csv_Agentic_retrieval", domain="medical"),
    'islamic': load_rag_system(collection_name="islamic_texts_Agentic_retrieval", domain="islamic"),
    'insurance': load_rag_system(collection_name="etiqa_Agentic_retrieval", domain="insurance")
}

# Store the loaded systems and llm on the app itself
# This allows the blueprint to access them using 'current_app'
app.rag_systems = rag_systems
app.llm = llm

# Register the API blueprint with your main Flask app
app.register_blueprint(api_bp)
logger.info(f"✅ API Blueprint registered. API endpoints are now available under /api")


# Check initialization status
logger.info("\n📊 SYSTEM STATUS:")
for domain, system in rag_systems.items():
    status = "✅ Ready" if system else "❌ Failed" 
    logger.info(f"   {domain}: {status}")


@app.route("/")
def homePage():
    # Clear all session history when visiting the home page
    session.pop('medical_history', None)
    session.pop('islamic_history', None)
    session.pop('insurance_history', None)
    session.pop('current_medical_document', None)
    return render_template("homePage.html")


@app.route("/medical", methods=["GET", "POST"])
def medical_page():
    # Use session for history and document context
    if request.method == "GET":
        # Load all latest data from session (or default to empty if not found)
        latest_response = session.pop('latest_medical_response', {}) # POP to clear it after one display
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")

        # Clear history only when a user first navigates (not on redirect)
        if not latest_response and 'medical_history' not in session:
            session.pop('current_medical_document', None)
        
        return render_template("medical_page.html", 
                               history=session.get('medical_history', []),
                               answer=answer,
                               thoughts=thoughts,
                               validation=validation,
                               source=source)
    
    # POST Request Logic
    answer, thoughts, validation, source = "", "", "", ""
    history = session.get('medical_history', [])
    current_medical_document = session.get('current_medical_document', "")
    
    
    try:
        query=standardize_query(request.form.get("query", ""))
        has_image = 'image' in request.files and request.files['image'].filename
        has_document = 'document' in request.files and request.files['document'].filename
        has_query = request.form.get("query") or request.form.get("question", "")
            
        logger.info(f"POST request received: has_image={has_image}, has_document={has_document}, has_query={has_query}")
            
        if has_document:
            # Scenario 3: Query + Document
            logger.info("Processing Scenario 3: Query + Document with Medical Swarm")
            #medical_chat_history = [] # Reset history for the new document
            file = request.files['document']
            try:
                # Store the new document text in the session
                document_text = file.read().decode("utf-8")
                session['current_medical_document'] = document_text
                current_medical_document = document_text # Use the new document for this turn
            except UnicodeDecodeError:
                answer = "Error: Could not decode the uploaded document. Please ensure it is a valid text or PDF file."
                logger.error("Scenario 3: Document decode error")
                thoughts = traceback.format_exc()
                  
            swarm_answer = run_medical_swarm(current_medical_document, query)
            answer = markdown_bold_to_html(swarm_answer)
                
            history.append(HumanMessage(content=f"[Document Uploaded] Query: '{query}'"))
            history.append(AIMessage(content=swarm_answer))
            thoughts = "Swarm analysis complete. The process is orchestrated and does not use the ReAct thought process. You can now ask follow-up questions."
            source= "Medical Swarm"
            validation = (True, "Swarm output generated.") # Swarm has its own validation logic
            
        elif has_image :
            #Scenario 1 
            logger.info("Processing Multimodal RAG: Query + Image")
            # --- Step 1 & 2: Image Setup & Vision Analysis ---
            file = request.files['image']
            upload_dir = "Uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, file.filename)
            
            try:
                file.save(image_path)
                file.close()
            
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")
                
            
                vision_prompt = f"Analyze this image and identify the main subject in a single, concise sentence. The user's query is: '{query}'"
                message = HumanMessage(content=[
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_data}"}
                ])
                vision_response = llm.invoke([message])
                visual_prediction = vision_response.content
                logger.info(f"Vision Prediction: {visual_prediction}")

                # --- Create an Enhanced Query ---
                enhanced_query = (
                    f'User Query: "{query}" '
                    f'Context from an image provided by the LLM: "{visual_prediction}" '
                    'Based on the user\'s query and the context from LLM, provide a comprehensive answer.'
                )
                logger.info(f"Enhanced query : {enhanced_query}")
            

                agent = rag_systems['medical']
                response_dict = agent.answer(enhanced_query, chat_history=history)
                answer, thoughts, validation, source = parse_agent_response(response_dict)
                history.append(HumanMessage(content=query))
                history.append(AIMessage(content=answer))
            
            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.info(f"Successfully deleted temporary image file: {image_path}")
                    except PermissionError as e:
                        logger.warning(f"Could not remove {image_path} after processing. "
                                       f"File may be locked by another process (e.g., antivirus). Error: {e}")
            
        elif query:
            # --- SCENARIO 2: TEXT-ONLY QUERY OR SWARM FOLLOW-UP ---
            history_for_agent = history
            if current_medical_document:
                logger.info("Processing Follow-up Query for Document")
                history_for_agent = [HumanMessage(content=f"We are discussing this document:\n{current_medical_document}")] + history
            else:
                logger.info("Processing Text RAG query for Medical domain")
            
            logger.info(f"Original Query: '{query}'")
            print(f"📚 Using chat history with {len(history)} previous messages to create standalone query")
            standalone_query = get_standalone_question(query, history_for_agent,llm)
            logger.info(f"Standalone Query: '{standalone_query}'")
            
            agent = rag_systems['medical']
            response_dict = agent.answer(standalone_query, chat_history=history_for_agent)
            answer, thoughts, validation, source = parse_agent_response(response_dict)

            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=answer))

        else:
            raise ValueError("No query or file provided.")
    except Exception as e:
        logger.error(f"Error on /medical page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
    
    # Save updated history and LATEST RESPONSE DATA back to the session
    session['medical_history'] = history
    session['latest_medical_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
                             
    logger.debug(f"Redirecting after saving latest response.")
    return redirect(url_for('medical_page'))

@app.route("/medical/clear")
def clear_medical_chat():
    session.pop('medical_history', None)
    session.pop('current_medical_document', None)
    logger.info("Medical chat history cleared.")
    return redirect(url_for('medical_page'))

@app.route("/islamic", methods=["GET", "POST"])
def islamic_page():
    #Use session
    
    if request.method == "GET":
        # Load all latest data from session (or default to empty if not found)
        latest_response = session.pop('latest_islamic_response', {}) # POP to clear it after one display
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")
        
        # Clear history only when a user first navigates (no latest_response and no current history)
        if not latest_response and 'islamic_history' not in session:
            session.pop('islamic_history', None)
        
        return render_template("islamic_page.html", 
                                history=session.get('islamic_history', []),
                                answer=answer,
                                thoughts=thoughts,
                                validation=validation,
                                source=source)
    
    # POST Request Logic
    answer, thoughts, validation, source = "", "", "", ""
    history = session.get('islamic_history', [])
    
    # This try/except block wraps the ENTIRE POST logic
    try:
        query = standardize_query(request.form.get("query", ""))
        has_image = 'image' in request.files and request.files['image'].filename
        
        final_query = query # Default to the original query
        
        if has_image:
            logger.info("Processing Multimodal RAG query for Islamic domain")
            
            file = request.files['image']
            
            upload_dir = "Uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, file.filename)
            
            # --- FIX 1: Wrap all file operations in try/finally for cleanup ---
            try:
                file.save(image_path)
                # --- FIX 2: Explicitly close the Flask file stream to release the lock ---
                file.close() 
                
                # Read the base64 data from the file just saved
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                
                # 'os.remove' is moved to the 'finally' block

                vision_prompt = f"Analyze this image's main subject. User's query is: '{query}'"
                message = HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}])
                visual_prediction = llm.invoke([message]).content

                enhanced_query = (
                    f'User Query: "{query}" '
                    f'Context from an image provided by the LLM: "{visual_prediction}" '
                    'Based on the user\'s query and the context from LLM, provide a comprehensive answer.'
                )
                logger.info(f"Create enchanced query : {enhanced_query}")
                
                # --- FIX 3: Assign the enhanced_query to final_query so it gets used ---
                final_query = enhanced_query 
            
            finally:
                # --- FIX 4: Robust cleanup logic ---
                # This runs even if the AI logic fails, and it won't crash
                # if the file is locked by another process.
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.info(f"Successfully cleaned up {image_path}")
                    except PermissionError as e:
                        logger.warning(f"Could not remove {image_path} after processing. "
                                       f"File may be locked by another process (e.g., antivirus). Error: {e}")
            
        elif query: # Only run text logic if there's a query and no image
            logger.info("Processing Text RAG query for Islamic domain")
            standalone_query = get_standalone_question(query, history,llm)
            logger.info(f"Original Query: '{query}'")
            print(f"📚 Using chat history with {len(history)} previous messages to create standalone query")
            logger.info(f"Standalone Query: '{standalone_query}'")
            final_query = standalone_query
            
        if not final_query: 
            raise ValueError("No query or file provided.")
        
        agent = rag_systems['islamic']
        response_dict = agent.answer(final_query, chat_history=history)
        answer, thoughts , validation, source = parse_agent_response(response_dict)
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))

    except Exception as e:
        logger.error(f"Error on /islamic page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
            
    # Save updated history and LATEST RESPONSE DATA back to the session
    session['islamic_history'] = history
    session['latest_islamic_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
                        
    logger.debug(f"Redirecting after saving latest response.")
    return redirect(url_for('islamic_page'))

@app.route("/islamic/clear")
def clear_islamic_chat():
    session.pop('islamic_history', None)
    logger.info("Islamic chat history cleared.")
    return redirect(url_for('islamic_page'))

@app.route("/insurance", methods=["GET", "POST"])
def insurance_page():
    if request.method == "GET" :
        latest_response = session.pop('latest_insurance_response',{})
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")
        
        if not latest_response and 'insurance_history' not in session:
            session.pop('insurance_history', None)
        
        return render_template("insurance_page.html", # You will need to create this HTML file
                                history=session.get('insurance_history', []),
                                answer=answer,
                                thoughts=thoughts,
                                validation=validation,
                                source=source)
    
    # POST Request Logic
    answer, thoughts, validation, source = "", "", "", ""
    history = session.get('insurance_history', [])
    
    try:
        query = standardize_query(request.form.get("query", ""))
        
        if query:
            logger.info("Processing Text RAG query for Insurance domain")
            standalone_query = get_standalone_question(query, history, llm)
            logger.info(f"Original Query: '{query}'")
            logger.info(f"Standalone Query: '{standalone_query}'")
            
            agent = rag_systems['insurance']
            response_dict = agent.answer(standalone_query, chat_history=history)
            answer, thoughts, validation, source = parse_agent_response(response_dict)
            
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=answer))
        else:
            raise ValueError("No query provided.")

    except Exception as e:
        logger.error(f"Error on /insurance page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
            
    session['insurance_history'] = history
    session['latest_insurance_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
                        
    logger.debug(f"Redirecting after saving latest response.")
    return redirect(url_for('insurance_page'))

@app.route("/insurance/clear")
def clear_insurance_chat():
    session.pop('insurance_history', None)
    logger.info("Insurance chat history cleared.")
    return redirect(url_for('insurance_page'))

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(debug=True, host="0.0.0.0", port=5000,use_reloader=False)