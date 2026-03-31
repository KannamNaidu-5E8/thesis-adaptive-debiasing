import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from src.config import OUTPUTS_PATH
from src.phase4_evaluation.evaluator import PipelineEvaluator

# Set academic plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def main():
    print("=== Starting Phase 4: Validation & Re-evaluation ===")
    
    results_csv = os.path.join(OUTPUTS_PATH, "adaptive_pipeline_results.csv")
    if not os.path.exists(results_csv):
        print(f"Error: Could not find {results_csv}. Please run scripts/run_adaptive.py first.")
        return
        
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} processed prompts for evaluation.")
    
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
    
    print("Calculating Fluency Metrics...")
    df['baseline_length'] = df['raw_baseline_response'].apply(evaluator.calculate_length)
    df['mitigated_length'] = df['final_mitigated_response'].apply(evaluator.calculate_length)
    df['baseline_ttr'] = df['raw_baseline_response'].apply(evaluator.calculate_ttr)
    df['mitigated_ttr'] = df['final_mitigated_response'].apply(evaluator.calculate_ttr)
    
    final_output_file = os.path.join(OUTPUTS_PATH, "final_ablation_results.csv")
    df.to_csv(final_output_file, index=False)
    
    # --- GENERATE CHARTS ---
    print("\nGenerating Charts...")
    
    # Chart 1: Diagnosis Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='diagnosis', palette='viridis', order=df['diagnosis'].value_counts().index)
    plt.title('Taxonomic Breakdown of LLM Political Bias', fontweight='bold')
    plt.xlabel('Diagnosed Bias Category')
    plt.ylabel('Number of Prompts')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "chart_1_diagnosis_distribution.png"), dpi=300)
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
        plt.savefig(os.path.join(OUTPUTS_PATH, "chart_2_jsd_reduction.png"), dpi=300)
        plt.close()

    print(f"Charts saved to {OUTPUTS_PATH}")
    print("\n=== FINAL METRICS ===")
    print(f"Average Initial JSD: {df['initial_jsd_score'].mean():.4f}")
    print(f"Average Final JSD:   {df['final_jsd_score'].mean():.4f}")
    print("======================\n")

if __name__ == "__main__":
    main()