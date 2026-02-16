from typing import Dict, List, Any, Optional, Tuple
from .base_agent import BaseAgent
from prompts.prompt_templates import RESPONSE_AGENT_PROMPTS

class ResponseAgent(BaseAgent):
    """
    Response Agent: Generates and validates responses based on selected hypothesis.
    Implements the third stage of the Metamind system.
    """
    
    def __init__(self, config: Dict[str, Any], llm_interface):
        """
        Initialize the Response Agent.
        
        Args:
            config: Configuration for the Response Agent
            llm_interface: Interface to the language model
        """
        super().__init__(config, llm_interface)
        self.beta = config.get("beta", 0.8)
        self.utility_threshold = config.get("utility_threshold", 0.9) 
        self.max_revisions = config.get("max_revisions", 3) 
    
    def process(self, selected_hypothesis: Dict[str, Any], user_input: str, 
               conversation_context: List[Dict[str, str]], social_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and validate a response based on the selected hypothesis.
        
        Args:
            selected_hypothesis: Selected hypothesis from the Domain Agent
            user_input: Current user utterance
            conversation_context: Previous conversation history
            social_memory: Social memory data
            
        Returns:
            Generated response with metadata
        """
        # Step 1: Generate initial response
        response = self._generate_response(selected_hypothesis, user_input, social_memory)
        
        # Step 2: Validate response
        validation_result = self._validate_response(response, selected_hypothesis, 
                                                  user_input, conversation_context, social_memory)
        
        # Step 3: Optimize response if needed
        revision_count = 0
        while validation_result.get("utility", 0) < self.utility_threshold and revision_count < self.max_revisions:
            print(f"[ResponseAgent] Utility {validation_result.get('utility', 0)} < {self.utility_threshold}. Optimizing response (Revision {revision_count + 1}).")
            response, validation_result = self._optimize_response(response, validation_result, 
                                                                selected_hypothesis, user_input, 
                                                                conversation_context, social_memory)
            revision_count += 1
        
        if validation_result.get("utility", 0) < self.utility_threshold:
            print(f"[ResponseAgent] Max revisions reached. Final utility {validation_result.get('utility',0)} still below threshold.")

        return {
            "response": response,
            "validation": validation_result,
            "revisions_done": revision_count
        }
    
    def _generate_response(self, hypothesis: Dict[str, Any], user_input: str, 
                         social_memory: Dict[str, Any]) -> str:
        """
        Generate a response based on the selected hypothesis.
        """
        prompt = self._format_prompt(
            RESPONSE_AGENT_PROMPTS["response_synthesis"],
            h_tilde_explanation=hypothesis.get("explanation", "N/A"),
            h_tilde_type=hypothesis.get("type", "Unknown"),
            # T=hypothesis.get("type", "Unknown"),
            M_t=str(social_memory),
            u_t=user_input
        )
        
        raw_response = self.llm.generate(prompt)
        response_data = self._parse_response_generation(raw_response)
        
        return response_data.get("response", raw_response) 
    
    def _validate_response(self, response: str, hypothesis: Dict[str, Any], 
                         user_input: str, conversation_context: List[Dict[str, str]], 
                         social_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the generated response.
        """
        formatted_context = self._format_conversation_context(conversation_context)
        
        prompt = self._format_prompt(
            RESPONSE_AGENT_PROMPTS["response_validation"],
            o_t=response,
            h_tilde_explanation=hypothesis.get("explanation", "N/A"),
            h_tilde_type=hypothesis.get("type", "Unknown"),
            # T=hypothesis.get("type", "Unknown"),
            M_t=str(social_memory),
            C_t=formatted_context,
            u_t=user_input,
            beta=self.beta
        )
        
        validation_raw_response = self.llm.generate(prompt)
        validation_results = self._parse_validation_results(validation_raw_response)
        
        return validation_results
    
    def _optimize_response(self, original_response: str, validation_result: Dict[str, Any], 
                         hypothesis: Dict[str, Any], user_input: str, 
                         conversation_context: List[Dict[str, str]], 
                         social_memory: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize the response if it doesn't meet the utility threshold.
        """
        critique = validation_result.get("critique", "No specific critique provided.")
        formatted_context = self._format_conversation_context(conversation_context)

        prompt = self._format_prompt(
            RESPONSE_AGENT_PROMPTS["response_optimization"],
            original_response=original_response,
            critique=critique,
            h_tilde_explanation=hypothesis.get("explanation", "N/A"),
            h_tilde_type=hypothesis.get("type", "Unknown"),
            # T=hypothesis.get("type", "Unknown"),
            u_t=user_input,
            C_t=formatted_context,
            M_t=str(social_memory)
        )

        optimized_raw_response = self.llm.generate(prompt)
        optimized_response_data = self._parse_response_generation(optimized_raw_response) 
        new_response = optimized_response_data.get("response", optimized_raw_response)

        # Re-validate the new response
        new_validation_result = self._validate_response(new_response, hypothesis, user_input, 
                                                      conversation_context, social_memory)
        
        return new_response, new_validation_result

    def _parse_response_generation(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response from the response generation/synthesis task.
        This is a placeholder. The actual format depends on LLM's output structure.
        It might return JSON, or just plain text.
        """
        return {"response": llm_response.strip()}

    def _parse_validation_results(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response from the response validation task.
        Expected to contain at least a 'utility' score and optionally 'critique', 'empathy', 'coherence'.
        Example LLM output: "Utility: 0.85\nCritique: A bit too formal.\nEmpathy: 0.9\nCoherence: 0.8"
        """
        results = {"utility": 0.0, "critique": "Parsing failed.", "empathy": 0.0, "coherence": 0.0}
        try:
            lines = llm_response.strip().split('\n')
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "utility":
                        results["utility"] = float(value)
                    elif key == "critique":
                        results["critique"] = value
                    elif key == "empathy":
                        results["empathy"] = float(value)
                    elif key == "coherence":
                        results["coherence"] = float(value)
            if "utility" not in [k.strip().lower() for k in results.keys() if isinstance(results[k], float)] and results["empathy"] > 0 and results["coherence"] > 0:
                 # Check if empathy and coherence were successfully parsed as floats
                if isinstance(results["empathy"], float) and isinstance(results["coherence"], float):
                    results["utility"] = (self.beta * results["empathy"]) + ((1 - self.beta) * results["coherence"])

        except Exception as e:
            print(f"[ResponseAgent] Error parsing validation results: {e}\nResponse: {llm_response}")
            results["utility"] = 0.1 
            results["critique"] = f"Error parsing validation: {e}"
        
        return results