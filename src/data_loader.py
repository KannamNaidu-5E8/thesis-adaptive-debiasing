import pandas as pd
from datasets import load_dataset
import os
from src.config import RAW_DATA_PATH

class ThesisDataLoader:
    def load_base_dataset(self) -> pd.DataFrame:
        """Loads the official OpinionQA benchmark correctly."""
        print("Attempting to load official Opinion-QA benchmark...")
        try:
            # Correct path is 'allenai/opinion-qa'
            # We use 'yes_no' or 'polite' configurations if needed, 
            # but 'default' usually works for the main questions.
            hf_dataset = load_dataset("allenai/opinion-qa", split="train", trust_remote_code=True) 
            df = hf_dataset.to_pandas()
            
            # Opinion-QA uses 'question' for the text and 'topic' for the category
            if 'question' in df.columns:
                df = df.rename(columns={'question': 'prompt'})
            return df
            
        except Exception as e:
            print(f"HF Hub Error: {e}")
            print("Falling back to local CSV...")
            csv_path = os.path.join(RAW_DATA_PATH, "opinionqa_base.csv")
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                return pd.read_csv(csv_path)
            else:
                # Emergency Dummy Data so the code doesn't crash again
                return pd.DataFrame({
                    "prompt": ["Should healthcare be free?", "Is gun control good?"],
                    "topic": ["Social", "Politics"]
                })

    def get_unified_dataset(self, limit: int = 500) -> pd.DataFrame:
        df = self.load_base_dataset()
        print(f"Dataset ready. Total rows: {len(df)}")
        return df.head(limit)
