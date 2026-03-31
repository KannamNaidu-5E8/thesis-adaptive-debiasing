from src.phase1_detection.judge_llm import ZeroShotJudge
from src.phase1_detection.jsd_calculator import JSDMetric

class PipelineEvaluator:
    """Re-evaluates mitigated text to calculate final fairness and fluency metrics."""
    
    def __init__(self):
        # Re-use Phase 1 tools to grade the final output
        self.judge = ZeroShotJudge()
        self.jsd_calc = JSDMetric()

    def evaluate_final_jsd(self, final_text: str) -> float:
        """Calculates the JSD score of the mitigated response."""
        new_distribution = self.judge.evaluate_response(final_text)
        return self.jsd_calc.calculate_divergence(new_distribution)

    def calculate_ttr(self, text: str) -> float:
        """Calculates Type-Token Ratio (TTR) as a proxy for vocabulary richness/fluency."""
        if not text or not isinstance(text, str):
            return 0.0
            
        tokens = text.lower().split()
        if len(tokens) == 0: 
            return 0.0
            
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    def calculate_length(self, text: str) -> int:
        """Calculates the word count of the response."""
        if not text or not isinstance(text, str):
            return 0
        return len(text.split())