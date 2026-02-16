from typing import Dict, List, Any, Tuple
import math # For log in information gain calculation
from .base_agent import BaseAgent
from prompts.prompt_templates import DOMAIN_AGENT_PROMPTS

class DomainAgent(BaseAgent):
    """
    Domain Agent: Refines hypotheses based on domain-specific rules and selects the best one.
    Implements the second stage of the Metamind system.
    """
    def __init__(self, config: Dict[str, Any], llm_interface: Any, social_memory_interface: Any):
        """
        Initialize the Domain Agent.
        
        Args:
            config: Configuration for the Domain Agent.
            llm_interface: Interface to the language model.
            social_memory_interface: Interface to social memory (for context in P_cond).
        """
        super().__init__(config, llm_interface)
        self.social_memory = social_memory_interface
        self.lambda_weight = config.get("lambda_weight", 0.6)
        self.domain_prompts = DOMAIN_AGENT_PROMPTS
        self.epsilon = 1e-9 # Small constant for log stability

    def process(self, hypotheses: List[Dict[str, Any]], user_input: str, 
                conversation_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Refine hypotheses and select the most appropriate one.
        
        Args:
            hypotheses: A list of candidate mental state hypotheses from ToMAgent.
            user_input: Current user utterance.
            conversation_context: Previous conversation history.
            
        Returns:
            The selected and refined hypothesis.
        """
        refined_hypotheses = []
        for h_i in hypotheses:
            # Step 1: Domain Constraint Refinement Task
            # This would involve specific prompts for cultural, ethical, role-based rules.
            refined_h_i = self._refine_hypothesis_with_constraints(h_i)
            refined_h_i["original_hypothesis_id"] = h_i.get("id")
            refined_hypotheses.append(refined_h_i)
        
        # Step 2: Hypothesis Selection 
        selected_hypothesis = self._select_hypothesis(refined_hypotheses, user_input, conversation_context)
        
        return selected_hypothesis

    def _refine_hypothesis_with_constraints(self, hypothesis: Dict[str, Any], domain_rules: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Refine a single hypothesis based on domain rules.
        (Corresponds to 'Domain Constraint Refinement Task' in details.tex)
        """
        # Placeholder for actual implementation using LLM and specific prompts for each rule type.
        prompt = self._format_prompt(
            self.domain_prompts["constraint_refinement"],
            h_i_explanation=hypothesis["explanation"],
            h_i_type=hypothesis["type"],
            h_i_id=hypothesis.get("id", "N/A"),
            domain_rule_type="Cultural",
            constraint_specifications=str(domain_rules.get("cultural", []))
        )
        response = self.llm.generate(prompt)
        refined_data = self._parse_refinement_response(response)
        hypothesis.update(refined_data) # Update hypothesis with refined explanation, etc.
        print(f"[DomainAgent] Refining hypothesis: {hypothesis.get('id')}")
        hypothesis["explanation"] = "(Refined by DomainAgent) " + hypothesis.get("explanation", "")
        return hypothesis

    def _select_hypothesis(self, refined_hypotheses: List[Dict[str, Any]], user_input: str, 
                           conversation_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Select the best hypothesis based on contextual plausibility and information gain.
        (Implements Algorithm 1 from details.tex)
        """
        if not refined_hypotheses:
            # Fallback or raise error if no hypotheses to select from
            return {"explanation": "No suitable hypothesis found.", "type": "Error", "score": -1}

        best_hypothesis = None
        max_score = -float('inf')
        
        formatted_context = self._format_conversation_context(conversation_context)
        social_memory_summary = str(self.social_memory.get_summary(user_id="default_user")) # Example

        for h_tilde_i in refined_hypotheses:
            # P_cond = P(~h_i | u_t, C_t, M_t)
            p_cond = self._get_conditional_probability(h_tilde_i, user_input, formatted_context, social_memory_summary)
            
            # P_prior = P(~h_i)
            p_prior = self._get_prior_probability(h_tilde_i)
            
            # Information Gain (IG_i)
            # IG_i = log(P_cond + epsilon) - log(P_prior + epsilon)
            ig_i = math.log(p_cond + self.epsilon) - math.log(p_prior + self.epsilon)
            
            # Composite score (s_i)
            # s_i = lambda * P_cond + (1-lambda) * IG_i
            score_i = (self.lambda_weight * p_cond) + ((1 - self.lambda_weight) * ig_i)
            
            h_tilde_i['score'] = score_i # Store score for reference
            h_tilde_i['p_cond'] = p_cond
            h_tilde_i['p_prior'] = p_prior
            h_tilde_i['ig'] = ig_i

            if score_i > max_score:
                max_score = score_i
                best_hypothesis = h_tilde_i
                
        return best_hypothesis if best_hypothesis else refined_hypotheses[0] # Fallback

    def _get_conditional_probability(self, hypothesis: Dict[str, Any], user_input: str, 
                                     formatted_context: str, social_memory_summary: str) -> float:
        """
        Estimate P(~h_i | u_t, C_t, M_t) using an LLM.
        This is a placeholder; actual implementation would involve a carefully crafted prompt
        that asks the LLM to score the plausibility of the hypothesis given the context.
        The LLM might return a score (e.g., 0-1) or a logit that can be normalized.
        """
        prompt = self._format_prompt(
            self.domain_prompts["conditional_probability"],
            h_tilde_i_explanation=hypothesis["explanation"],
            h_tilde_i_type=hypothesis["type"],
            u_t=user_input,
            C_t=formatted_context,
            M_t=social_memory_summary
        )
        response = self.llm.generate(prompt)
        probability = self._parse_probability_response(response)
        return probability

    def _get_prior_probability(self, hypothesis: Dict[str, Any]) -> float:
        """
        Estimate P(~h_i) using an LLM or a simpler heuristic.
        This represents the general likelihood of the hypothesis irrespective of the current context.
        """
        prompt = self._format_prompt(
            self.domain_prompts["prior_probability"],
            h_tilde_i_explanation=hypothesis["explanation"],
            h_tilde_i_type=hypothesis["type"]
        )
        response = self.llm.generate(prompt)
        probability = self._parse_probability_response(response)
        return probability
    def _parse_probability_response(self, llm_response: str) -> float:
        """
        Parse LLM response that's expected to be a probability or score.
        """
        try:
            return float(llm_response.strip())
        except ValueError:
            return 0.5 

    def _parse_refinement_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response from the 'Domain Constraint Refinement Task'.
        Expected format (from details.tex):
        - Original Hypothesis: ...
        - Revised Hypothesis: ...
        - Modification Log: ...
        """
        return {
            "explanation": "Parsed refined explanation: " + llm_response,
            "modification_log": {"parsed_log": "details..."}
        }