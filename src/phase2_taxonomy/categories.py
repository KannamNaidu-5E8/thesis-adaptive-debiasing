from enum import Enum

class BiasCategory(Enum):
    """Defines the strict taxonomic categories for political bias."""
    
    PARTISAN_SKEW = "Partisan Skew"
    SAFETY_REFUSAL = "Safety Refusal"
    NEUTRAL_FAIR = "Neutral / Fair"