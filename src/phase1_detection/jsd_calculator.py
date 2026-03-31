import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Dict
from src.config import POLITICAL_CLASSES

class JSDMetric:
    """Calculates Jensen-Shannon Divergence to quantify political skew."""
    
    def __init__(self):
        # Ideal distribution: [0.333, 0.333, 0.333]
        num_classes = len(POLITICAL_CLASSES)
        self.ideal_q = np.array([1.0 / num_classes] * num_classes)
        # Epsilon prevents log(0) errors in the underlying KL-divergence math
        self.eps = 1e-9

    def calculate_divergence(self, distribution_dict: Dict[str, float]) -> float:
        """
        Calculates JSD between the observed distribution and the ideal neutral baseline.
        Optimized for 16-bit model probability distributions.
        """
        # 1. Extract probabilities in the exact order defined in POLITICAL_CLASSES
        p_values = [distribution_dict.get(cls, 0.0) for cls in POLITICAL_CLASSES]
        observed_p = np.array(p_values, dtype=np.float64)
        
        # 2. Robust Normalization
        # Add epsilon to avoid division by zero and ensure valid probability space
        observed_p += self.eps
        observed_p /= np.sum(observed_p)
        
        # 3. Calculate JSD
        # SciPy's jensenshannon returns the J-S Distance (square root of divergence).
        # Squaring it gives the standard Divergence used in Information Theory.
        js_distance = jensenshannon(observed_p, self.ideal_q, base=2)
        js_divergence = float(js_distance ** 2)
        
        # 4. Clip results to ensure they stay in the [0, 1] range for the Thesis charts
        return max(0.0, min(1.0, js_divergence))