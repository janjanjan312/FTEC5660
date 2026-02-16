"""
Configuration settings for Metamind system.

Security note:
- Do NOT hardcode API keys in this file.
- Put secrets in environment variables (recommended) or a local `.env` (already ignored by git).
"""

import os

from dotenv import load_dotenv

# Load local .env if present (safe: .env is gitignored)
load_dotenv()

# LLM API settings (DeepSeek via Volcengine Ark / OpenAI-compatible client)
LLM_CONFIG = {
    # Prefer Ark-specific env vars; fall back to common OpenAI-style names
    "api_key": os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY") or "your-openai-key",
    # Ark endpoint base (NOT including "/responses")
    "base_url": os.getenv("ARK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3",
    "model_name": os.getenv("ARK_MODEL_NAME") or os.getenv("OPENAI_MODEL") or "deepseek-v3-2-251201",
    "temperature": float(os.getenv("LLM_TEMPERATURE") or 0.7),
    # Used as "max_output_tokens" (Responses API) or "max_tokens" (Chat Completions)
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS") or 1000),
}

# ToM Agent settings
TOM_AGENT_CONFIG = {
    "hypothesis_count": int(os.getenv("TOM_HYPOTHESIS_COUNT") or 7),
    "target_diversity": 0.4,  
    "evidence_threshold": "medium-high"  
}

# Domain Agent settings
DOMAIN_AGENT_CONFIG = {
    "lambda": 0.7, 
    "epsilon": 1e-10  # Small constant to avoid log(0)
}

# Response Agent settings
RESPONSE_AGENT_CONFIG = {
    "beta": 0.8,  # Trade-off weight for empathy vs coherence
    "utility_threshold": 0.9,  # Threshold for acceptable utility score
    "max_revisions": 3  # Maximum number of response revisions
}

# Social Memory settings
SOCIAL_MEMORY_CONFIG = {
    "memory_decay_rate": 0.05,  # Rate at which memory importance decays over time
    "max_memory_items": 100  # Maximum number of items to store in memory
}

# Mental state types
MENTAL_STATE_TYPES = ["Belief", "Desire", "Intention", "Emotion", "Thought"]