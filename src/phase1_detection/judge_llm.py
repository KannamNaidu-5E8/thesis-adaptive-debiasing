from transformers import pipeline
from typing import Dict
import torch
from src.config import JUDGE_MODEL_ID, POLITICAL_CLASSES

class ZeroShotJudge:
    """Evaluates the political leaning of generated text using an NLI model."""
    
    def __init__(self):
        print(f"Initializing Judge Model: {JUDGE_MODEL_ID}...")
        
        # Optimization for A40-2Q:
        # We use device=0 to keep it on the GPU for speed,
        # but we use torch_dtype=torch.float16 to reduce its VRAM footprint.
        # Inside ZeroShotJudge class
        self.classifier = pipeline(
                "zero-shot-classification", 
                model=JUDGE_MODEL_ID, 
                device=-1  # FORCE TO CPU (-1)
            )

    def evaluate_response(self, text: str) -> Dict[str, float]:
        """Returns a probability distribution across defined political classes."""
        if not text or not text.strip():
            # If the model outputs nothing, return a perfectly balanced distribution
            return {cls: 1.0 / len(POLITICAL_CLASSES) for cls in POLITICAL_CLASSES}

        # Inference is now much faster in FP16 on the GPU
        result = self.classifier(text, POLITICAL_CLASSES, multi_label=False)
        
        # Map labels back to their specific scores
        distribution = {
            label: score for label, score in zip(result['labels'], result['scores'])
        }
        return distribution