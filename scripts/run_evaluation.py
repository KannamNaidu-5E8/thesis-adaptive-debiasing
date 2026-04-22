import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Path Injection (Crucial for Kaggle) ---
# This forces Python to recognize the current directory as a package root,
# allowing 'from src.config' to work properly.
sys.path.append(os.getcwd())

# --- Imports ---
from src.config import OUTPUTS_PATH
from src.phase4_evaluation.evaluator import PipelineEvaluator

# Set academic plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def main():
    print("=== Starting Phase 4: Validation & Re-evaluation ===")
    print(f"DEBUG: Using output directory: {OUTPUTS_PATH}")
    
    # 1. Verify existence of the input file
    results_csv = os.path.join(OUTPUTS_PATH, "adaptive_pipeline_results.csv")
    
    if not os.path.exists(results_csv):
        print(f"Error: Could not find '{results_csv}'.")
        print("Please ensure the previous step ran successfully and saved the file there.")
        print(f"Current contents of output folder: {os.listdir(OUTPUTS_PATH) if os.path.exists(OUTPUTS_PATH) else 'Folder not found'}")
        return
        
    # 2. Load and process
    df = pd.read_csv(results_csv)
    print(f"Successfully loaded {len(df)} processed prompts for evaluation.")
    
    evaluator = PipelineEvaluator()
    final_jsd_scores = []
    
    print("\nCalculating Final JSD Scores...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row['mitigation_applied'] == "PASS_THROUGH":
            final_jsd_scores.append(row['initial_jsd_score'])
        else:
            new_jsd = evaluator.evaluate_final_jsd(str(row['final_mitigated_response']))
            final_jsd_scores.append(new_jsd)
            
    df['final_jsd_score'] = final_jsd_scores
    df['jsd_improvement'] = df['initial_jsd_score'] - df['final_jsd_score']
    
    # 3. Save ablation results
    final_output_file = os.path.join(OUTPUTS_PATH, "final_ablation_results.csv")
    df.to_csv(final_output_file, index=False)
    print(f"Saved results to: {final_output_file}")
    
    # 4. Generate Charts
    print("\nGenerating Charts...")
    
    # Chart 1: Diagnosis Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='diagnosis', palette='viridis', order=df['diagnosis'].value_counts().index)
    plt.title('Taxonomic Breakdown of LLM Political Bias', fontweight='bold')
    plt.xlabel('Diagnosed Bias Category')
    plt.ylabel('Number of Prompts')
    plt.tight_layout()
    chart1_path = os.path.join(OUTPUTS_PATH, "chart_1_diagnosis_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    
    # Chart 2: JSD Reduction
    intervened_df = df[df['mitigation_applied'] != 'PASS_THROUGH']
    if not intervened_df.empty:
        plot_data = pd.DataFrame({
            'Metric': ['Baseline LLM (Raw)'] * len(intervened_df) + ['Adaptive Framework (Mitigated)'] * len(intervened_df),
            'JSD Score': list(intervened_df['initial_jsd_score']) + list(intervened_df['final_jsd_score'])
        })
        
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=plot_data, x='Metric', y='JSD Score', palette='coolwarm', inner="quartile")
        plt.title('Reduction in Political Bias via Adaptive Mitigation', fontweight='bold')
        plt.ylabel('Jensen-Shannon Divergence (Lower = More Neutral)')
        plt.axhline(y=0.15, color='r', linestyle='--', label='Bias Threshold')
        plt.legend()
        plt.tight_layout()
        chart2_path = os.path.join(OUTPUTS_PATH, "chart_2_jsd_reduction.png")
        plt.savefig(chart2_path, dpi=300)
        plt.close()

    print(f"Charts saved to {OUTPUTS_PATH}")
    print("\n=== FINAL METRICS ===")
    print(f"Average Initial JSD: {df['initial_jsd_score'].mean():.4f}")
    print(f"Average Final JSD:   {df['final_jsd_score'].mean():.4f}")
    print("======================\n")

if __name__ == "__main__":
    main()
