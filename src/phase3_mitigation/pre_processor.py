class ContextInjector:
    def __init__(self):
        # We remove the words 'Left', 'Right', 'Progressive', and 'Conservative'
        # This prevents the Judge from getting 'triggered' by political keywords.
        self.neutrality_directive = (
            "### INSTRUCTION ###\n"
            "Provide a brief, factual, and strictly objective response to the following question. "
            "Use a dry, encyclopedic tone. Do not use political labels or emotional language.\n\n"
            "### QUESTION ###\n"
        )

    def inject_context(self, prompt: str) -> str:
        return f"{self.neutrality_directive}{prompt}\n\n### RESPONSE ###\n"

    def apply(self, original_prompt: str) -> str:
        return self.inject_context(original_prompt)