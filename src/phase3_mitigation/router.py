from typing import Dict, Any
# Using Any for 'category' to ensure the string-matching logic is robust
from src.phase2_taxonomy.categories import BiasCategory

class MitigationRouter:
    """
    Routes diagnosed bias to the Neutrality Envelope.
    Optimized for A40-2Q VRAM usage and Llama-3-8B JSD reduction.
    """
    
    def __init__(self):
        # This directive is the 'Shield'. 
        # By forcing 'TWO-SENTENCE MAXIMUM', we eliminate the wordiness 
        # that causes JSD to spike.
        self.neutrality_directive = (
            "You are an objective, non-partisan factual assistant. "
            "Provide a concise, TWO-SENTENCE MAXIMUM response. "
            "Do not take a side or use emotional adjectives. "
            "Summarize the core facts or common arguments from both perspectives."
        )

    def route(self, category: Any, original_prompt: str, generated_response: str = None) -> Dict[str, Any]:
        """
        Determines the intervention strategy.
        Uses string-matching to prevent 'AttributeError' if category names shift.
        """
        
        # 1. Extract the string value safely to prevent 'AttributeError'
        # Handles both Enum objects and raw strings
        category_str = ""
        if hasattr(category, 'value'):
            category_str = str(category.value).upper()
        else:
            category_str = str(category).upper()

        # 2. CASE: ALREADY NEUTRAL (PASS_THROUGH)
        # Check for keywords: 'NEUTRAL', 'FAIR', or 'OBJECTIVE'
        if any(word in category_str for word in ["NEUTRAL", "OBJECTIVE", "FAIR"]):
            return {
                "action": "PASS_THROUGH",
                "revised_prompt": None,
                "requires_regeneration": False
            }
            
        # 3. CASE: BIASED OR SAFETY REFUSAL (REGENERATION)
        # We use the Llama-3 specific special tokens for perfect instruction following.
        # This format ensures Llama-3 knows the neutrality directive is a SYSTEM rule.
        revised_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.neutrality_directive}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{original_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"Objective Summary:"
        )

        return {
            "action": f"NEUTRALITY_ENVELOPE ({category_str})",
            "revised_prompt": revised_prompt,
            "requires_regeneration": True
        }