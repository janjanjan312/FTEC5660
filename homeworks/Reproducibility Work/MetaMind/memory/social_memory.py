import datetime
import json
import logging
from typing import Dict, Any, List, Optional

from llm_interface.base_llm import BaseLLM 
from utils.helpers import parse_json_from_string 

logger = logging.getLogger(__name__)

class SocialMemory:
    """
    Manages social memory (M_t) by leveraging an LLM for interpretation and summarization.
    The LLM helps in extracting preferences, emotional markers, and generating summaries from interactions.
    """
    def __init__(self, llm_interface: BaseLLM, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Social Memory with an LLM interface.

        Args:
            llm_interface: An instance of a class that implements BaseLLM.
            config: Optional configuration for the social memory.
        """
        self.llm = llm_interface
        self.config = config if config else {}
        self.user_memory: Dict[str, Dict[str, Any]] = {}
        # Define prompts here or load them from a dedicated prompt module/config
        self.prompts = {
            "extract_preference": "Given the following user utterance and conversation context, identify any user preferences (e.g., likes, dislikes, interests, communication style). Format the output as a JSON object with preference keys and values. If no clear preferences are stated or implied, return an empty JSON object.\n\nUser Utterance: {utterance}\nConversation Context: {context}\n\nJSON Output:",
            "extract_emotion": "Analyze the user's utterance for emotional content. Identify the primary emotion and its intensity (0.0 to 1.0). Consider the conversation context. Format as a JSON object: {{\"emotion\": \"type\", \"intensity\": value, \"evidence\": \"textual_evidence\"}}. If no clear emotion is detected, return an empty JSON object.\n\nUser Utterance: {utterance}\nConversation Context: {context}\n\nJSON Output:",
            "summarize_interaction_for_memory": "Summarize the key points of this interaction for long-term memory. Focus on user needs, agent actions, and outcomes. Keep it concise. \n\nUser: {user_utterance}\nAgent: {agent_response}\n\nSummary:",
            "generate_memory_summary_for_prompt": "Based on the stored user preferences, recent emotional markers, and interaction history, generate a concise summary of the user's social memory. This summary will be used to inform an AI agent's responses. Highlight key aspects relevant for personalization and empathetic interaction.\n\nPreferences: {preferences}\nRecent Emotions: {emotions}\nInteraction History: {history}\n\nConcise Summary for Agent:"
        }

    def _ensure_user_memory_exists(self, user_id: str):
        """
        Ensure that the memory structure for a given user_id exists.
        """
        if user_id not in self.user_memory:
            self.user_memory[user_id] = {
                "preferences": {}, 
                "emotional_markers": [], 
                "interaction_history": [] 
            }

    def update_memory_from_interaction(self, user_id: str, user_utterance: str, agent_response: str, conversation_context: List[str]):
        """
        Processes a new interaction, using LLM to extract preferences, emotions,
        and to summarize the interaction for storage.
        """
        self._ensure_user_memory_exists(user_id)
        context_str = "\n".join(conversation_context[-5:]) # Use recent context

        # 1. Extract Preferences using LLM
        try:
            pref_prompt = self.prompts["extract_preference"].format(utterance=user_utterance, context=context_str)
            llm_response_pref = self.llm.generate(pref_prompt, max_tokens=100) # Adjust max_tokens as needed
            extracted_prefs = parse_json_from_string(llm_response_pref, logger)
            if extracted_prefs:
                for key, value in extracted_prefs.items():
                    self.update_preference(user_id, key, value, from_llm=True)
        except Exception as e:
            logger.error(f"[SocialMemory] LLM failed to extract preferences for user {user_id}: {e}")

        # 2. Extract Emotional Markers using LLM
        try:
            emo_prompt = self.prompts["extract_emotion"].format(utterance=user_utterance, context=context_str)
            llm_response_emo = self.llm.generate(emo_prompt, max_tokens=100)
            extracted_emotion = parse_json_from_string(llm_response_emo, logger)
            if extracted_emotion and extracted_emotion.get("emotion"):
                self.add_emotional_marker(
                    user_id,
                    emotion=extracted_emotion["emotion"],
                    intensity=float(extracted_emotion.get("intensity", 0.5)),
                    context=user_utterance, # Or more specific context from LLM
                    evidence=extracted_emotion.get("evidence"),
                    from_llm=True
                )
        except Exception as e:
            logger.error(f"[SocialMemory] LLM failed to extract emotions for user {user_id}: {e}")

        # 3. Summarize Interaction for Memory using LLM
        try:
            summary_prompt = self.prompts["summarize_interaction_for_memory"].format(
                user_utterance=user_utterance, 
                agent_response=agent_response
            )
            interaction_summary_text = self.llm.generate(summary_prompt, max_tokens=150).strip()
            if interaction_summary_text:
                self.add_interaction(user_id, interaction_summary_text, from_llm=True)
        except Exception as e:
            logger.error(f"[SocialMemory] LLM failed to summarize interaction for user {user_id}: {e}")

    def update_preference(self, user_id: str, key: str, value: Any, from_llm: bool = False):
        self._ensure_user_memory_exists(user_id)
        self.user_memory[user_id]["preferences"][key] = value
        source = "LLM" if from_llm else "direct"
        logger.info(f"[SocialMemory] Updated preference for user {user_id} via {source}: {key} = {value}")

    def get_preference(self, user_id: str, key: str) -> Optional[Any]:
        self._ensure_user_memory_exists(user_id)
        return self.user_memory[user_id]["preferences"].get(key)

    def add_emotional_marker(self, user_id: str, emotion: str, intensity: float = 1.0, 
                             context: Optional[str] = None, evidence: Optional[str] = None, from_llm: bool = False):
        self._ensure_user_memory_exists(user_id)
        marker = {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "context": context if context else "N/A",
            "evidence": evidence if evidence else "N/A",
            "source": "LLM" if from_llm else "direct"
        }
        self.user_memory[user_id]["emotional_markers"].append(marker)
        logger.info(f"[SocialMemory] Added emotional marker for user {user_id} via {marker['source']}: {emotion}")

    def get_recent_emotional_markers(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        self._ensure_user_memory_exists(user_id)
        sorted_markers = sorted(
            self.user_memory[user_id]["emotional_markers"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return sorted_markers[:limit]

    def add_interaction(self, user_id: str, summary: str, tags: Optional[List[str]] = None, from_llm: bool = False):
        self._ensure_user_memory_exists(user_id)
        interaction = {
            "summary": summary,
            "tags": tags if tags else [],
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "LLM" if from_llm else "direct"
        }
        self.user_memory[user_id]["interaction_history"].append(interaction)
        source = "LLM" if from_llm else "direct"
        logger.info(f"[SocialMemory] Added interaction summary for user {user_id} via {source}")

    def get_interaction_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._ensure_user_memory_exists(user_id)
        sorted_interactions = sorted(
            self.user_memory[user_id]["interaction_history"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return sorted_interactions[:limit]

    def get_summary(self, user_id: str, max_preferences: int = 5, max_emotions: int = 3, max_interactions: int = 3) -> str:
        """
        Generates a consolidated summary of a user's social memory using an LLM for use in prompts.
        """
        self._ensure_user_memory_exists(user_id)
        
        preferences_summary_obj = dict(list(self.user_memory[user_id]["preferences"].items())[:max_preferences])
        emotions_summary_obj = self.get_recent_emotional_markers(user_id, limit=max_emotions)
        interactions_summary_obj = self.get_interaction_history(user_id, limit=max_interactions)

        # Convert objects to string representations for the prompt
        prefs_str = json.dumps(preferences_summary_obj, indent=2)
        emos_str = json.dumps(emotions_summary_obj, indent=2)
        hist_str = json.dumps(interactions_summary_obj, indent=2)

        prompt = self.prompts["generate_memory_summary_for_prompt"].format(
            preferences=prefs_str,
            emotions=emos_str,
            history=hist_str
        )
        
        try:
            llm_summary = self.llm.generate(prompt, max_tokens=200) # Adjust max_tokens
            return llm_summary.strip()
        except Exception as e:
            logger.error(f"[SocialMemory] LLM failed to generate prompt summary for user {user_id}: {e}")
            return f"User Preferences: {prefs_str}\nRecent Emotions: {emos_str}\nRecent Interactions: {hist_str}"

    def get_all_data_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.user_memory.get(user_id)

    def clear_user_memory(self, user_id: str):
        if user_id in self.user_memory:
            del self.user_memory[user_id]
            logger.info(f"[SocialMemory] Cleared memory for user {user_id}")
        else:
            logger.info(f"[SocialMemory] No memory found for user {user_id} to clear.")