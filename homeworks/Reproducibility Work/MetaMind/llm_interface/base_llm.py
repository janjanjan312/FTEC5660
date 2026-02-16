from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model interfaces.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM interface.

        Args:
            config: Configuration dictionary for the LLM (e.g., API keys, model names).
        """
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional keyword arguments for the LLM API (e.g., temperature, max_tokens).

        Returns:
            The generated text string.
        """
        pass
