import logging
from flask import Blueprint, request, jsonify, current_app
from langchain_core.messages import HumanMessage, AIMessage

# Import your project's functions
from src.utils import get_standalone_question
from src.medical_swarm import run_medical_swarm

# --- 1. Create the Blueprint ---
# This is a 'mini' Flask app that will be joined to your main app.
# All routes defined here will be prefixed with /api
api_bp = Blueprint('api', __name__, url_prefix='/api')

logger = logging.getLogger(__name__)

# --- Helper Function ---
def convert_chat_history(history_json: list) -> list:
    """Converts JSON chat history from the request to LangChain message objects."""
    messages = []
    for msg in history_json:
        if msg.get('role') == "human":
            messages.append(HumanMessage(content=msg.get('content')))
        elif msg.get('role') == "ai":
            messages.append(AIMessage(content=msg.get('content')))
    return messages

# --- 2. Define API Endpoints ---

@api_bp.route('/query_rag', methods=['POST'])
def handle_rag_query():
    """
    Endpoint for the main RAG agent (Medical or Islamic).
    Handles text queries and chat history.
    Expects JSON: { "query": "...", "domain": "...", "chat_history": [...] }
    """
    logger.info("[API] RAG query received")
    data = request.json
    query = data.get('query')
    domain = data.get('domain')
    chat_history_json = data.get('chat_history', [])

    if not query or not domain:
        return jsonify({"error": "Missing 'query' or 'domain'"}), 400

    # --- Access the loaded systems from the main Flask app ---
    # We use 'current_app' to access the objects loaded in app.py
    try:
        rag_systems = current_app.rag_systems
        llm = current_app.llm
    except AttributeError:
        logger.error("[API] RAG systems or LLM not found on current_app.")
        return jsonify({"error": "Server not initialized. Check main app logs."}), 500

    if domain not in rag_systems:
        return jsonify({"error": f"Invalid domain. Must be one of: {list(rag_systems.keys())}"}), 400

    agent = rag_systems[domain]
    if not agent:
        return jsonify({"error": f"RAG system for domain '{domain}' is not loaded."}), 500

    # --- Run the same logic as your app.py ---
    try:
        history_objects = convert_chat_history(chat_history_json)
        standalone_query = get_standalone_question(query, history_objects, llm)
        
        logger.info(f"[API] Original query: '{query}' -> Standalone: '{standalone_query}'")
        
        response_dict = agent.answer(standalone_query, chat_history=history_objects)
        
        # Add the standalone_query to the response for clarity
        response_dict['standalone_query'] = standalone_query
        
        # 'answer' key is already HTML, which is fine.
        # 'validation' is a tuple [bool, str], which JSON can handle.
        return jsonify(response_dict)
    
    except Exception as e:
        logger.error(f"[API] Error during RAG execution: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500

@api_bp.route('/analyze_document', methods=['POST'])
def handle_swarm_analysis():
    """
    Endpoint for the Medical Swarm to analyze an uploaded document text.
    Expects JSON: { "query": "...", "document_text": "..." }
    """
    logger.info("[API] Medical Swarm request received.")
    data = request.json
    document_text = data.get('document_text')
    query = data.get('query')

    if not document_text or not query:
        return jsonify({"error": "Missing 'document_text' or 'query'"}), 400

    # --- Run the Swarm ---
    try:
        swarm_answer = run_medical_swarm(
            document_text=document_text,
            initial_query=query
        )
        
        # The swarm answer is a markdown string.
        return jsonify({"answer": swarm_answer})
    
    except Exception as e:
        logger.error(f"[API] Error during swarm execution: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500

@api_bp.route('/', methods=['GET'])
def api_root():
    """Root endpoint for the API blueprint to confirm it's running."""
    return jsonify({"message": "RAG & Swarm API is running. Use /query_rag or /analyze_document."})
