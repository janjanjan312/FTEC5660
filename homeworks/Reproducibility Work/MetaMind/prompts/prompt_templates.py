# Prompts for Stage 1: Theory-of-Mind (ToM) Agent
TOM_AGENT_PROMPTS = {
    "contextual_analysis": """
Contextual Analysis Task
Input:
- User's current statement (u_t): {u_t}
- Conversational context (C_t):
{C_t}
Objective:
Generate 3-5 commonsense interpretations of the user’s unstated needs by:
1. Identifying key semantic triggers in the utterance.
2. Mapping these triggers to plausible psychosocial motivations.
3. Considering cultural and linguistic norms for indirect communication.
Output Format (one interpretation per line):
- Interpretation: [Explanation] (Contextual Support: [Relevant C_t Excerpt])
""",
    "memory_integration": """
Memory Integration Task
Input:
- Proposed hypothesis: {selected_common_sense_interpretation}
- Social memory database (M_t):
{M_t}
Objective: Assess the hypothesis's alignment with past interactions and user profile.
Output Format:
"Hypothesis [ID] shows [strong/medium/weak] memory alignment (Score: [1-5]). Key corroborations: [List relevant memory items or observations]."
""",
    "mental_state_typology": """
Mental State Typology Task
Input:
- Utterance (u_t): {u_t}
- Top Hypothesis (interpretation): {interpretation}
- Memory Correlations (findings): {findings}
Objective: Classify the primary mental state type for the hypothesis.
Classification Markers:
- Belief: Cognitive representations of reality.
- Desire: Preferences or goal states.
- Intention: Action-oriented plans.
- Emotion: Affective states.
- Thought: Conscious reasoning processes.
Output Format:
- Primary Marker: [Belief/Desire/Intention/Emotion/Thought] (Confidence: [Percentage])
  Rationale: [Psychological justification]
""",
    "mental_state_hypothesis_generation": """  # Renamed from mental_state_space_planning for clarity
Generate Mental State Hypothesis Task

Objective: Based on the user's input, conversation context, and social memory, generate a plausible mental state hypothesis. Focus on the mental state type: {T_focus}.

Inputs:
- User Input (u_t): {u_t}
- Conversational Context (C_t):
{C_t}
- Social Memory Summary (M_t):
{M_t}
- Target Mental State Type (T_focus): {T_focus}

Instructions:
1.  Analyze the provided inputs.
2.  Formulate a hypothesis about the user's mental state, specifically aligning with the target type {T_focus}.
3.  The hypothesis should be a concise explanation.

Output Format (Strictly follow this format):
Type: [The {T_focus} type, e.g., Belief, Desire, Intention, Emotion, Thought]
Description: [A 1-2 sentence natural language explanation of the hypothesis]
"""
}

# Prompts for Stage 2: Domain Agent
DOMAIN_AGENT_PROMPTS = {
    "constraint_refinement": """
Domain Constraint Refinement Task

Objective: Refine the given hypothesis based on domain-specific rules and constraints.

Inputs:
- Original Hypothesis Explanation (h_i_explanation): {h_i_explanation}
- Original Hypothesis Type (h_i_type): {h_i_type}
- Original Hypothesis ID (h_i_id): {h_i_id}

Instructions:
1.  Identify elements in '{h_i_explanation}' that may conflict with or need adjustment based on the '{domain_rule_type}' constraints specified in '{constraint_specifications}'.
2.  Re-interpret and revise the hypothesis explanation to be compliant and appropriate.
3.  If the core nature of the hypothesis changes, suggest a revised type.

Output Format (Strictly follow this format):
Revised Hypothesis Explanation: [Your socially compliant and refined interpretation of the hypothesis]
Revised Hypothesis Type: [Original type, or a new type if significantly altered, e.g., Belief, Desire]
Modification Log: [Briefly describe changes made, e.g., "Adjusted tone for formality based on cultural rule X", "Removed ethically sensitive implication Y"]
""",
    "conditional_probability": """
Estimate Contextual Plausibility (P_cond) Task

Objective: Assess the contextual plausibility of the given refined hypothesis based on the user's prompt, conversation context, and social memory. Output a single probability score between 0.0 (very unlikely) and 1.0 (very likely).

Inputs:
- Refined Hypothesis Explanation (h_tilde_i_explanation): {h_tilde_i_explanation}
- Refined Hypothesis Type (h_tilde_i_type): {h_tilde_i_type}
- User Prompt (u_t): {u_t}
- Conversational Context (C_t):
{C_t}
- Social Memory Summary (M_t):
{M_t}

Output Format (Return only the score):
[A single float number between 0.0 and 1.0]
""",
    "prior_probability": """
Estimate Prior Plausibility (P_prior) Task

Objective: Assess the general, context-independent plausibility of this type of hypothesis. Output a single probability score between 0.0 (very unlikely) and 1.0 (very likely).

Inputs:
- Refined Hypothesis Explanation (h_tilde_i_explanation): {h_tilde_i_explanation}
- Refined Hypothesis Type (h_tilde_i_type): {h_tilde_i_type}

Output Format (Return only the score):
[A single float number between 0.0 and 1.0]
"""
}

# Prompts for Stage 3: Response Agent
RESPONSE_AGENT_PROMPTS = {
    "response_synthesis": """
Response Synthesis Task

Objective: Generate a natural language response that is empathetic, coherent, and aligned with the selected hypothesis, user input, and social memory.

Inputs:
- Selected Hypothesis Explanation (h_tilde_explanation): {h_tilde_explanation}
- Selected Hypothesis Type (h_tilde_type): {h_tilde_type}
- User Input (u_t): {u_t}
- Social Memory Summary (M_t):
{M_t}

Output Format (Return only the generated response text):
[Your empathetic and coherent response here]
""",
    "response_validation": """
Response Validation Task

Objective: Evaluate the generated response based on empathy and coherence, given the context. Provide scores and a brief critique.

Inputs:
- Generated Response (o_t): {o_t}
- Selected Hypothesis Explanation (h_tilde_explanation): {h_tilde_explanation}
- Selected Hypothesis Type (h_tilde_type): {h_tilde_type}
- User Input (u_t): {u_t}
- Conversational Context (C_t):
{C_t}
- Social Memory Summary (M_t):
{M_t}
- Beta (Weight for Empathy vs. Coherence): {beta}

Task: Evaluate the response and provide:
1.  Empathy Score (0.0-1.0): How well the response resonates with the user's inferred emotional or cognitive state ({h_tilde_explanation}).
2.  Coherence Score (0.0-1.0): Consistency with conversational context ({C_t}) and task constraints.
3.  Overall Utility Score: Calculated as {beta} * Empathy + (1-{beta}) * Coherence. (You don't need to calculate this, just provide Empathy and Coherence scores).
4.  Critique: Brief textual feedback on why the response is good or how it could be improved.

Output Format (Strictly follow this format, each on a new line):
Empathy: [A single float number between 0.0 and 1.0]
Coherence: [A single float number between 0.0 and 1.0]
Utility: [A single float number between 0.0 and 1.0, calculated as {beta} * Empathy + (1-{beta}) * Coherence]
Critique: [Your brief textual feedback]
""",
    "response_optimization": """
Response Optimization Task

Objective: Revise the original response based on the critique to improve its utility, empathy, and coherence.

Inputs:
- Original Response: {original_response}
- Critique of Original Response: {critique}
- Selected Hypothesis Explanation (h_tilde_explanation): {h_tilde_explanation}
- Selected Hypothesis Type (h_tilde_type): {h_tilde_type}
- User Input (u_t): {u_t}
- Conversational Context (C_t):
{C_t}
- Social Memory Summary (M_t):
{M_t}

Task: Revise the original response based on the critique.

Output Format (Return only the optimized response text):
[Your optimized response here]
"""
}