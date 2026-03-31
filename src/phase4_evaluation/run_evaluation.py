import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from src.config import OUTPUTS_PATH
from src.phase4_evaluation.evaluator import PipelineEvaluator

# Set academic plotting style for NIT Agartala Thesis
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def main():
    print("=== Starting Phase 4: Validation & Re-evaluation ===")
    
    # 1. Load the results from Phase 3
    results_csv = os.path.join(OUTPUTS_PATH, "adaptive_pipeline_results.csv")
    if not os.path.exists(results_csv):
        print(f"Error: Could not find {results_csv}. Please run scripts/run_adaptive.py first.")
        return
        
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} processed prompts for evaluation.")
    
    evaluator = PipelineEvaluator()
    final_jsd_scores = []
    
    # 2. Re-calculate JSD for the final outputs
    print("\nCalculating Final JSD Scores...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row['mitigation_applied'] == "PASS_THROUGH":
            # If no mitigation was applied, the score hasn't changed
            final_jsd_scores.append(row['initial_jsd_score'])
        else:
            # Evaluate the newly generated, mitigated text
            new_jsd = evaluator.evaluate_final_jsd(str(row['final_mitigated_response']))
            final_jsd_scores.append(new_jsd)
            
    df['final_jsd_score'] = final_jsd_scores
    df['jsd_improvement'] = df['initial_jsd_score'] - df['final_jsd_score']
    
    # 3. Calculate Fluency Metrics (Length & TTR)
    print("Calculating Fluency Metrics...")
    df['baseline_length'] = df['raw_baseline_response'].apply(evaluator.calculate_length)
    df['mitigated_length'] = df['final_mitigated_response'].apply(evaluator.calculate_length)
    df['baseline_ttr'] = df['raw_baseline_response'].apply(evaluator.calculate_ttr)
    df['mitigated_ttr'] = df['final_mitigated_response'].apply(evaluator.calculate_ttr)
    
    # Save the fully evaluated dataset
    final_output_file = os.path.join(OUTPUTS_PATH, "final_ablation_results.csv")
    df.to_csv(final_output_file, index=False)
    
    # --- 4. GENERATE ACADEMIC CHARTS ---
    print("\nGenerating Thesis Charts...")
    
    # Chart 1: The Diagnosis Distribution
    plt.figure(figsize=(9, 6))
    ax = sns.countplot(
        data=df, 
        x='diagnosis', 
        hue='diagnosis', 
        palette='viridis', 
        legend=False,
        order=df['diagnosis'].value_counts().index
    )
    plt.title('Taxonomic Breakdown of LLM Political Bias', fontweight='bold', fontsize=14)
    plt.xlabel('Diagnosed Bias Category', fontsize=12)
    plt.ylabel('Number of Prompts', fontsize=12)
    
    # Add data labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "chart_1_diagnosis_distribution.png"), dpi=300)
    plt.close()
    
    # Chart 2: Ablation Efficacy (The JSD Drop)
    # Filter for cases where the framework actively intervened (Mitigated prompts)
    intervened_df = df[df['mitigation_applied'] != 'PASS_THROUGH']
    
    if not intervened_df.empty:
        # Prepare long-form data for Seaborn
        plot_data = pd.DataFrame({
            'Metric': ['Raw LLM'] * len(intervened_df) + ['Adaptive Mitigated'] * len(intervened_df),
            'JSD Score': list(intervened_df['initial_jsd_score']) + list(intervened_df['final_jsd_score'])
        })
        
        plt.figure(figsize=(8, 6))
        ax_violin = sns.violinplot(
            data=plot_data, 
            x='Metric', 
            y='JSD Score', 
            hue='Metric', 
            palette='coolwarm', 
            inner="quartile", 
            legend=False
        )
        
        # DYNAMIC TITLE: Automatically updates N to match your dataset size
        plt.title(f'Bias Reduction via Adaptive Mitigation (N={len(df)})', fontweight='bold', fontsize=14)
        plt.ylabel('JSD Score (Lower = More Neutral)', fontsize=12)
        plt.xlabel('')
        
        # Add a threshold line for "High Bias" reference
        plt.axhline(y=0.10, color='red', linestyle='--', alpha=0.6, label='High Bias Threshold')
        
        # Fix the legend so it only shows the threshold line cleanly
        handles, labels = ax_violin.get_legend_handles_labels()
        plt.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_PATH, "chart_2_jsd_reduction.png"), dpi=300)
        plt.close()

    print(f"Professional charts saved to {OUTPUTS_PATH}")
    
    # 5. Print Final Summary Table for the Terminal
    print("\n" + "="*40)
    print("        NIT AGARTALA: FINAL ABLATION METRICS")
    print("="*40)
    print(f"Avg Initial JSD (Raw LLM):     {df['initial_jsd_score'].mean():.4f}")
    print(f"Avg Final JSD (Mitigated):     {df['final_jsd_score'].mean():.4f}")
    print(f"Avg JSD Improvement:           {df['jsd_improvement'].mean():.4f}")
    print("-" * 40)
    print(f"Mean Baseline TTR (Richness):  {df['baseline_ttr'].mean():.3f}")
    print(f"Mean Mitigated TTR (Richness): {df['mitigated_ttr'].mean():.3f}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()