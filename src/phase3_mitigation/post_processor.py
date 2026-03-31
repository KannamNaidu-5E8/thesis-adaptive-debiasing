class ConstitutionalCorrector:
    """Post-processes biased or refused outputs by forcing the LLM to self-correct."""
    
    def __init__(self):
        self.critique_template = (
            "System Directive: Your previous response to the prompt was flagged for either "
            "unjustified political censorship (refusing to answer) or extreme partisan bias. "
            "Rewrite your response to directly answer the prompt using purely objective, "
            "factual statements without relying on safety evasion or ideological attacks.\n\n"
            "Original Prompt: {prompt}\n"
            "Flagged Response: {response}\n\n"
            "Corrected Objective Response:"
        )

    def apply(self, original_prompt: str, flagged_response: str) -> str:
        """Formats the critique prompt to trigger self-correction."""
        return self.critique_template.format(
            prompt=original_prompt,
            response=flagged_response
        )