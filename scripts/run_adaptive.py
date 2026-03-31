import pandas as pd
import os
from tqdm import tqdm

# Import from our src directory
from src.data_loader import ThesisDataLoader
from src.pipeline import AdaptiveDebiasPipeline
from src.config import OUTPUTS_PATH

def main():
    print("=== Initializing Adaptive Taxonomic Debiasing Framework ===")
    
    # 1. Load Data
    loader = ThesisDataLoader()
    dataset_df = loader.get_unified_dataset()
    
    # 2. Initialize Pipeline 
    pipeline = AdaptiveDebiasPipeline()
    
    # 3. Process the batch
    results = []
    print("\nStarting batch evaluation...")
    
    for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        prompt = row['prompt']
        topic = row['topic']
        
        try:
            # Run the prompt through the full framework
            lifecycle_data = pipeline.process_prompt(prompt)
            lifecycle_data['topic'] = topic
            results.append(lifecycle_data)
            print("Processed prompt: ", prompt)
        except Exception as e:
            print(f"\nError processing prompt '{prompt}': {e}")
            
    # 4. Save the results for Phase 4
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUTS_PATH, "adaptive_pipeline_results.csv")
    results_df.to_csv(output_file, index=False)
    
    print(f"\n=== Execution Complete! ===")
    print(f"Results saved to: {output_file}")
    
    # Print a quick summary of what the system diagnosed
    print("\nDiagnosis Summary:")
    print(results_df['diagnosis'].value_counts())

if __name__ == "__main__":
    main()