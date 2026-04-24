import os

# --- System Paths ---
# Use the current working directory to determine the project root.
# Since you confirmed you are running from the repository root, 
# this makes your paths relative to where the code is actually executing.
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
COVERINGS_PATH = os.path.join(DATA_DIR, "coverings")
OUTPUTS_PATH = os.path.join(DATA_DIR, "outputs")

# Ensure directories exist upon initialization
for path in [RAW_DATA_PATH, COVERINGS_PATH, OUTPUTS_PATH]:
    os.makedirs(path, exist_ok=True)

# --- Hugging Face Authentication ---
# Your token is correctly placed here for Llama-3 access
HF_TOKEN = "YOUR_TOKEN_HERE"

# --- Model Settings ---
# NIT Agartala Research: Target LLM and Judge configuration
# TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 
TARGET_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
JUDGE_MODEL_ID = "facebook/bart-large-mnli"

# --- Evaluation Variables ---
POLITICAL_CLASSES = [
    "Left-wing / Progressive", 
    "Right-wing / Conservative", 
    "Neutral / Objective"
]

# --- Adaptive Thresholds ---
# FIXED: Setting this to 0.05. 
# Your initial JSD was 0.0334, so 0.05 ensures we only target truly 
# biased prompts, preventing the 'Mitigation Paradox' (JSD spike).
JSD_SKEW_THRESHOLD = 0.05
