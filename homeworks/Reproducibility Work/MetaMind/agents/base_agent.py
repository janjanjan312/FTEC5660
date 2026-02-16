from typing import Dict, Any, List

class BaseAgent:
    """
    Base class for all agents in the Metamind system.
    """
    def __init__(self, config: Dict[str, Any], llm_interface: Any):
        """
        Initialize the Base Agent.
        
        Args:
            config: Configuration for the agent.
            llm_interface: Interface to the language model.
        """
        self.config = config
        self.llm = llm_interface

    def _format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with given keyword arguments.
        
        Args:
            template: The prompt template string.
            **kwargs: Keyword arguments to fill into the template.
            
        Returns:
            The formatted prompt string.
        """
        return template.format(**kwargs)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response string into a structured format.
        This method should be overridden by subclasses if specific parsing is needed.
        
        Args:
            response: The raw response string from the LLM.
            
        Returns:
            A dictionary containing the parsed data.
        """
        # Basic parsing, assuming response is a simple string or can be directly used.
        # Subclasses might need more sophisticated parsing (e.g., JSON, XML).
        return {"text": response}

    def _format_conversation_context(self, conversation_context: List[Dict[str, str]]) -> str:
        """
        Format the conversation context for inclusion in prompts.

        Args:
            conversation_context: A list of dictionaries, where each dictionary
                                  represents a turn with "speaker" and "utterance".

        Returns:
            A formatted string representing the conversation history.
        """
        if not conversation_context:
            return "No previous conversation history."
        
        formatted_history = []
        for turn in conversation_context:
            speaker = turn.get("speaker", "Unknown")
            utterance = turn.get("utterance", "")
            formatted_history.append(f"{speaker}: {utterance}")
        return "\n".join(formatted_history)

    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method for the agent.
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Each agent must implement its own process method.")