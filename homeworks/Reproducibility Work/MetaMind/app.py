from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import logging


from main import MetamindApplication, LLM_CONFIG 

app = Flask(__name__)
CORS(app) 



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if not LLM_CONFIG["api_key"] or LLM_CONFIG["api_key"] == "your_api_key_here":
    logger.error("OpenAI API key is not configured in config.py. Flask app might not work correctly.")
    
    

metamind_app = None
try:
    metamind_app = MetamindApplication()
    logger.info("MetamindApplication initialized successfully for Flask app.")
except Exception as e:
    logger.error(f"Failed to initialize MetamindApplication: {e}", exc_info=True)
    
    



conversation_history = [] 

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', history=conversation_history)

@app.route('/process', methods=['POST'])
def process_input():
    """Handles user input, processes it through Metamind, and returns results."""
    if not metamind_app:
        logger.error("MetamindApplication not initialized. Cannot process request.")
        return jsonify({"error": "Application backend not ready."}), 500

    try:
        data = request.get_json()
        user_utterance = data.get('utterance')
        
        if not user_utterance:
            return jsonify({"error": "No utterance provided"}), 400

        
        
        
        context_strings = []
        for entry in conversation_history[-5:]: 
            context_strings.append(f"User: {entry['user']}")
            context_strings.append(f"Metamind: {entry['metamind']}")

        logger.info(f"Received utterance: '{user_utterance}' with context_strings: {context_strings}")
        
        
        
        processing_results = metamind_app.process_user_input(user_utterance, context_strings)
        
        
        current_interaction = {
            'user': user_utterance,
            'metamind': processing_results.get('final_response', 'Error processing response.'),
            'details': processing_results 
        }
        conversation_history.append(current_interaction)
        
        logger.info(f"Processed results: {processing_results}")
        return jsonify(current_interaction) 

    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    
    app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000/