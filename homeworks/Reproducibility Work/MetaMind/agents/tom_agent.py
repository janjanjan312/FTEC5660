from typing import Dict, List, Any, Optional, Tuple
from .base_agent import BaseAgent
from prompts.prompt_templates import TOM_AGENT_PROMPTS

class ToMAgent(BaseAgent):
    """
    Theory-of-Mind (ToM) Agent: Generates mental state hypotheses.
    Implements the first stage of the Metamind system.
    """
    def __init__(self, config: Dict[str, Any], llm_interface: Any, social_memory_interface: Any):
        """
        Initialize the ToM Agent.
        
        Args:
            config: Configuration for the ToM Agent.
            llm_interface: Interface to the language model.
            social_memory_interface: Interface to the social memory.
        """
        super().__init__(config, llm_interface)
        self.social_memory = social_memory_interface
        # Backward/forward compatible config key:
        # - config.py uses "hypothesis_count"
        # - some older code may use "hypothesis_count_k"
        self.hypothesis_count_k = config.get("hypothesis_count", config.get("hypothesis_count_k", 7))
        # Initialize self.tom_prompts here if _contextual_analysis is to be used
        self.tom_prompts = TOM_AGENT_PROMPTS 

    def process(self, user_input: str, conversation_context: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Generate a set of mental state hypotheses based on user input and context.
        
        Args:
            user_input: Current user utterance.
            conversation_context: Previous conversation history.
            
        Returns:
            A list of candidate mental state hypotheses.
        """
        # commonsense_interpretations = self._contextual_analysis(user_input, conversation_context)
        hypotheses = []
        formatted_context = self._format_conversation_context(conversation_context)
        social_memory_summary = str(self.social_memory.get_summary(user_id="default_user")) # Or a more structured summary

        for i in range(self.hypothesis_count_k):
            current_focus_type = self._get_next_hypothesis_type_focus(i)
            prompt = self._format_prompt(
                template=self.tom_prompts["mental_state_hypothesis_generation"], # Use the specific prompt key
                u_t=user_input,
                C_t=formatted_context,
                M_t=social_memory_summary,
                T_focus=current_focus_type 
            )
            
            raw_hypothesis_data = self.llm.generate(prompt)
            parsed_hypothesis = self._parse_hypothesis_generation(raw_hypothesis_data, i, expected_type=current_focus_type)
            if parsed_hypothesis:
                hypotheses.append(parsed_hypothesis)
        
        return hypotheses

    def _contextual_analysis(self, user_input: str, conversation_context: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Perform contextual analysis to generate initial interpretations.
        (Corresponds to 'Contextual Analysis Task' in details.tex)
        """

        prompt = self._format_prompt(self.tom_prompts["contextual_analysis"], u_t=user_input, C_t=self._format_conversation_context(conversation_context))
        response = self.llm.generate(prompt)
        interpretations = self._parse_contextual_analysis_response(response)
        return interpretations

    def _get_next_hypothesis_type_focus(self, index: int) -> str:
        """
        Helper to cycle through hypothesis types for diversity.
        """
        types = ["Belief", "Desire", "Intention", "Emotion", "Thought"]
        return types[index % len(types)]

    def _parse_hypothesis_generation(self, llm_response: str, hypothesis_index: int, expected_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            lines = llm_response.strip().split('\n')
            hypothesis_data = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "type":
                        hypothesis_data["type"] = value
                    elif key == "description":
                        hypothesis_data["explanation"] = value # Matching 'explanation' used elsewhere
            
            # Ensure essential fields are present
            if "explanation" not in hypothesis_data:
                # If parsing fails to find description, use the whole response as description
                # and try to infer type or use expected_type
                hypothesis_data["explanation"] = llm_response.strip()
            
            if "type" not in hypothesis_data and expected_type:
                hypothesis_data["type"] = expected_type
            elif "type" not in hypothesis_data:
                 hypothesis_data["type"] = "Unknown"

            return {
                "id": f"hyp_{hypothesis_index + 1}",
                "explanation": hypothesis_data.get("explanation"),
                "type": hypothesis_data.get("type"),
                "evidential_basis": {"linguistic_signals": "N/A", "contextual_drivers": "N/A", "memory_anchors": "N/A"} # Placeholder
            }
        except Exception as e:
            print(f"[ToMAgent] Error parsing hypothesis: {e}\nResponse: {llm_response}")
            return None