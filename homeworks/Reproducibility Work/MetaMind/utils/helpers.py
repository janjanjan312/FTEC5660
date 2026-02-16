import logging
import json
from typing import Optional, Dict, Any

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a basic logger.

    Args:
        name: The name of the logger.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers if already configured
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def parse_json_from_string(json_string: str, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """
    Safely parse a JSON string, which might be embedded in a larger text
    or have imperfections (e.g., from LLM output).

    Args:
        json_string: The string potentially containing JSON.
        logger: Optional logger instance for error reporting.

    Returns:
        A dictionary if parsing is successful, None otherwise.
    """
    if not json_string:
        return None

    # Try to find JSON within ```json ... ``` blocks or as a whole
    try:

        return json.loads(json_string)
    except json.JSONDecodeError:
        # Attempt 2: Look for JSON within markdown-style code blocks
        try:
            start_index = json_string.find('{')
            end_index = json_string.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                potential_json = json_string[start_index : end_index + 1]
                return json.loads(potential_json)
        except json.JSONDecodeError as e_inner:
            if logger:
                logger.error(f"Failed to parse JSON from string. Error: {e_inner}. String: '{json_string[:200]}...'", exc_info=False)
            else:
                print(f"Error: Failed to parse JSON from string. Error: {e_inner}. String: '{json_string[:200]}...'", flush=True)
    return None