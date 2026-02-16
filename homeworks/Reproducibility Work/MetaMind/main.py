import logging
from config import LLM_CONFIG, TOM_AGENT_CONFIG, DOMAIN_AGENT_CONFIG, RESPONSE_AGENT_CONFIG, SOCIAL_MEMORY_CONFIG, MENTAL_STATE_TYPES
from llm_interface import OpenAILLM 
from memory import SocialMemory
from agents import ToMAgent, DomainAgent, ResponseAgent
from utils.helpers import setup_logger, parse_json_from_string

logger = setup_logger("MetamindApp", level=logging.INFO) 

class MetamindApplication:
    def __init__(self):
        logger.info("Initializing Metamind Application...")
        
        # 1. Initialize LLM Interface
        self.llm = OpenAILLM(LLM_CONFIG)
        logger.info(f"LLM Interface initialized with chat model: {LLM_CONFIG['model_name']}")

        # 2. Initialize Social Memory
        self.social_memory = SocialMemory(llm_interface=self.llm)
        logger.info("Social Memory initialized.")

        # 3. Initialize Agents
        self.tom_agent = ToMAgent(llm_interface=self.llm, social_memory_interface=self.social_memory, config=TOM_AGENT_CONFIG)
        self.domain_agent = DomainAgent(llm_interface=self.llm, social_memory_interface=self.social_memory, config=DOMAIN_AGENT_CONFIG)
        self.response_agent = ResponseAgent(llm_interface=self.llm, config=RESPONSE_AGENT_CONFIG)
        logger.info("Agents (ToM, Domain, Response) initialized.")

    def process_user_input(self, user_utterance: str, conversation_context: list[str]) -> dict:
        """
        Processes a single user utterance through the Metamind pipeline.
        Returns a dictionary containing all intermediate results and the final response.
        """
        logger.info(f"Processing user input: '{user_utterance}'")
        logger.debug(f"Current conversation context: {conversation_context}")

        results = {
            "user_utterance": user_utterance,
            "conversation_context_input": conversation_context,
            "tom_agent_hypotheses": [],
            "domain_agent_selected_hypothesis": None,
            "response_agent_details": None,
            "final_response": "I'm having trouble understanding that right now. Could you try rephrasing?",
            "social_memory_updated_summary": None,
            "error_message": None
        }

        # Stage 1: Theory-of-Mind (ToM) Agent
        try:
            hypotheses_H_t = self.tom_agent.process(user_input=user_utterance, conversation_context=conversation_context)
            results["tom_agent_hypotheses"] = hypotheses_H_t
            
            if not hypotheses_H_t:
                logger.warning("ToM Agent did not generate any hypotheses.")
                results["error_message"] = "ToM Agent did not generate any hypotheses."
                # No early return, let's see if we can proceed or provide a generic response later
            else:
                logger.info(f"ToM Agent generated {len(hypotheses_H_t)} hypotheses.")
                for i, h in enumerate(hypotheses_H_t):
                    logger.debug(f"  Hypothesis {i+1}: Type='{h.get('type')}', Desc='{h.get('explanation')}'")
        except Exception as e:
            logger.error(f"Error in ToM Agent: {e}", exc_info=True)
            results["error_message"] = f"Error in ToM Agent: {str(e)}"
            # Potentially return results here if ToM is critical and failed
            # For now, we'll let it try to continue or use default/error values
            hypotheses_H_t = [] # Ensure it's an empty list if an error occurred

        # Stage 2: Domain Agent
        selected_hypothesis = None
        if hypotheses_H_t: # Only proceed if ToM agent produced something
            try:
                selected_hypothesis = self.domain_agent.process(
                    hypotheses=hypotheses_H_t,
                    user_input=user_utterance,
                    conversation_context=conversation_context
                )
                results["domain_agent_selected_hypothesis"] = selected_hypothesis
                
                if not selected_hypothesis or selected_hypothesis.get("type") == "Error":
                    logger.warning("Domain Agent did not select a suitable hypothesis.")
                    # Fallback to the first ToM hypothesis if available
                    if hypotheses_H_t:
                        selected_hypothesis = hypotheses_H_t[0]
                        results["domain_agent_selected_hypothesis"] = selected_hypothesis # Update with fallback
                        logger.info("Falling back to the first ToM hypothesis for response generation.")
                    else:
                        results["error_message"] = (results["error_message"] + "; " if results["error_message"] else "") + "Domain Agent failed and no ToM fallback."
                        selected_hypothesis = None # Ensure it's None
                else:
                    logger.info(f"Domain Agent selected hypothesis (Score={selected_hypothesis.get('score', 0.0):.2f}): Type='{selected_hypothesis.get('type')}', Desc='{selected_hypothesis.get('explanation')}'")
            except Exception as e:
                logger.error(f"Error in Domain Agent: {e}", exc_info=True)
                results["error_message"] = (results["error_message"] + "; " if results["error_message"] else "") + f"Error in Domain Agent: {str(e)}"
                if hypotheses_H_t: # Fallback if domain agent errors out
                    selected_hypothesis = hypotheses_H_t[0]
                    results["domain_agent_selected_hypothesis"] = selected_hypothesis
                    logger.info("Error in Domain Agent, falling back to the first ToM hypothesis.")
                else:
                    selected_hypothesis = None
        elif not results["error_message"]: # If ToM was successful but returned empty, note it
             results["error_message"] = "ToM Agent returned no hypotheses, Domain Agent skipped."

        # Stage 3: Response Agent
        if selected_hypothesis: # Only proceed if we have a hypothesis
            try:
                social_memory_data_for_response = self.social_memory.get_summary(user_id="default_user") 
                if not isinstance(social_memory_data_for_response, dict):
                    social_memory_data_for_response = {"summary": str(social_memory_data_for_response)}

                response_details = self.response_agent.process(
                    selected_hypothesis=selected_hypothesis,
                    user_input=user_utterance,
                    conversation_context=conversation_context,
                    social_memory=social_memory_data_for_response 
                )
                results["response_agent_details"] = response_details
                results["final_response"] = response_details.get("response", "I'm not sure how to respond to that.")
                logger.info(f"Response Agent generated: '{results['final_response']}' (Revisions: {response_details.get('revisions_done',0)})")
            except Exception as e:
                logger.error(f"Error in Response Agent: {e}", exc_info=True)
                results["error_message"] = (results["error_message"] + "; " if results["error_message"] else "") + f"Error in Response Agent: {str(e)}"
                results["final_response"] = "Sorry, I encountered an issue while formulating my response."
        elif not results["error_message"]:
            results["error_message"] = "No hypothesis selected, Response Agent skipped."
            results["final_response"] = "I'm still trying to process that. Can you provide more details?"
        
        # Update social memory regardless of errors in agent chain, if user input was processed
        try:
            self.social_memory.update_memory_from_interaction("default_user", user_utterance, results["final_response"], conversation_context)
            results["social_memory_updated_summary"] = self.social_memory.get_summary(user_id="default_user")
            if not isinstance(results["social_memory_updated_summary"], dict):
                 results["social_memory_updated_summary"] = {"summary": str(results["social_memory_updated_summary"])}
        except Exception as e:
            logger.error(f"Error updating social memory: {e}", exc_info=True)
            results["error_message"] = (results["error_message"] + "; " if results["error_message"] else "") + f"Error updating social memory: {str(e)}"

        return results

def run_interactive_session(app: MetamindApplication):
    """
    Runs a simple interactive command-line session.
    """
    logger.info("Starting interactive Metamind session. Type 'quit' to exit.")
    conversation_history = [] # Stores (user_utterance, agent_response) tuples
    current_context_strings = [] # Stores just the string form for context passing

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                logger.info("Exiting interactive session.")
                break
            
            if not user_input.strip():
                continue

            context_to_pass = current_context_strings[-10:]

            agent_response = app.process_user_input(user_input, context_to_pass)
            print(f"Metamind: {agent_response}")

            # Update history
            conversation_history.append((user_input, agent_response))
            current_context_strings.append(f"User: {user_input}")
            current_context_strings.append(f"Metamind: {agent_response}")

        except KeyboardInterrupt:
            logger.info("Session interrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"An error occurred during the session: {e}", exc_info=True)
            print("Sorry, an unexpected error occurred. Please try again.")

if __name__ == "__main__":
    if not LLM_CONFIG.get("api_key") or LLM_CONFIG.get("api_key") == "your-openai-key":
        logger.error("LLM API key is not configured. Please set ARK_API_KEY/OPENAI_API_KEY (or update config.py).")
        print("ERROR: LLM API key not configured. Please set ARK_API_KEY/OPENAI_API_KEY or update 'config.py'.")
        exit(1)
    
    metamind_app = MetamindApplication()
    # run_interactive_session(metamind_app) # We will replace this with Flask app
    # Example of how to use the modified process_user_input:
    sample_context = ["User: Hello there!", "Metamind: Hi! How can I help you today?"]
    output_data = metamind_app.process_user_input("Tell me a joke.", sample_context)
    import json
    print(json.dumps(output_data, indent=2))