import pandas as pd
import os
from tqdm import tqdm
import time
import torch

# Import from our src directory
from src.data_loader import ThesisDataLoader
from src.pipeline import AdaptiveDebiasPipeline
from src.config import OUTPUTS_PATH

def main():
    print("=== Initializing Adaptive Taxonomic Debiasing Framework ===")
    
    # 1. Initialize Loader & Pipeline
    loader = ThesisDataLoader()
    
    # TIP: For your final M.Tech run, change limit=5 to limit=None to run the whole set
    dataset_df = loader.get_unified_dataset(limit=None)
    
    # Clear CUDA cache before starting the heavy load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    pipeline = AdaptiveDebiasPipeline()
    results = []

    print("\n--- NIT Agartala Research: Starting Batch Evaluation ---")
    print(f"Target Hardware: 2GB Dedicated VRAM / 16GB Shared VRAM")
    
    # Use tqdm for a professional progress bar
    for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        start_time = time.time()
        try:
            # Process the prompt through the 4-Phase Framework
            res = pipeline.process_prompt(row['prompt'])
            
            # Add metadata for your Thesis Analysis
            res['topic'] = row.get('topic', 'General')
            res['execution_time_sec'] = round(time.time() - start_time, 2)
            
            # Extract final JSD score if it was mitigated, else keep initial
            # This ensures your final JSD column is accurate for evaluation
            res['final_jsd_score'] = res.get('initial_jsd_score') # Default
            
            results.append(res)
            
        except Exception as e:
            # Vital for 2GB VRAM stability: Log error and move to next prompt
            print(f"\n[ERROR] Row {index} failed: {e}")
            # Optional: Short sleep to let the GPU 'cool down' or clear cache
            time.sleep(2)
            continue

    # 2. Safety Check
    if not results:
        print("\nFATAL ERROR: No results generated. Check your src/pipeline.py for errors.")
        return

    # 3. Save Data
    results_df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    
    # Professional timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M")
    output_file = os.path.join(OUTPUTS_PATH, f"adaptive_results_{timestamp}.csv")
    
    results_df.to_csv(output_file, index=False)
    
    print(f"\n=== Execution Complete! ===")
    print(f"Data saved for Phase 4 Evaluation: {output_file}")
    
    # 4. Professional Summary for your Thesis Log
    print("\n" + "="*40)
    print("        NIT AGARTALA: FINAL BATCH SUMMARY")
    print("="*40)
    
    if 'diagnosis' in results_df.columns:
        print("\n[1] Bias Diagnosis Distribution:")
        print(results_df['diagnosis'].value_counts())
    
    if 'initial_jsd_score' in results_df.columns:
        avg_init = results_df['initial_jsd_score'].mean()
        # Note: True Final JSD calculation happens in Phase 4 (run_evaluation.py)
        # But we show the Baseline here for immediate confirmation.
        print(f"\n[2] Average Baseline JSD: {avg_init:.4f}")
        
    print(f"\n[3] Mitigation Summary:")
    mitigation_counts = results_df['mitigation_applied'].value_counts()
    print(mitigation_counts)
        
    print(f"\n[4] Computational Efficiency:")
    avg_time = results_df['execution_time_sec'].mean()
    print(f"Average Processing Time: {avg_time:.2f}s per prompt")
    print("="*40)

if __name__ == "__main__":
    main()