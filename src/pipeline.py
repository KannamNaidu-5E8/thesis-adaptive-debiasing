from typing import Dict, Any
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# Import custom research modules
from src.config import TARGET_MODEL_ID, HF_TOKEN, JSD_SKEW_THRESHOLD
from src.phase1_detection.judge_llm import ZeroShotJudge
from src.phase1_detection.jsd_calculator import JSDMetric
from src.phase2_taxonomy.classifier import TaxonomyClassifier
from src.phase3_mitigation.router import MitigationRouter

class TargetLLM:
    def __init__(self):
        print(f"--- NIT Agartala Research: Initializing {TARGET_MODEL_ID} ---")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Vital for A40
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True 
        )        

        effective_token = os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else HF_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, token=effective_token)
        
        # FIX 1: Use 'device_map' to force GPU 0
        # FIX 2: Set 'low_cpu_mem_usage=False' to stop CPU from hogging the process
        self.model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID,
            token=effective_token,
            device_map={"": 0},           
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,      
            attn_implementation="sdpa"     
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # FIX 3: Changed 'outputs' to 'output_tokens' to match your decoding logic
            output_tokens = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,           
                do_sample=False,              
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_len = inputs['input_ids'].shape[-1]
        new_tokens = output_tokens[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up memory after every prompt
        del inputs, output_tokens
        torch.cuda.empty_cache()
        gc.collect()
        
        return response

class AdaptiveDebiasPipeline:
    """The master orchestrator for the Diagnostic-Classification-Mitigation framework."""
    
    def __init__(self):
        # Initializing the sub-components
        self.target_llm = TargetLLM()
        self.judge = ZeroShotJudge()
        self.jsd_calc = JSDMetric()
        self.classifier = TaxonomyClassifier(jsd_threshold=JSD_SKEW_THRESHOLD) 
        self.router = MitigationRouter()

    def process_prompt(self, original_prompt: str) -> Dict[str, Any]:
        """Runs the 4-phase adaptive loop on a single prompt."""
        
        # PHASE 0: Baseline Generation
        raw_response = self.target_llm.generate(original_prompt)
        
        # PHASE 1: Detection (NLI Judge + JSD)
        distribution = self.judge.evaluate_response(raw_response)
        jsd_score = self.jsd_calc.calculate_divergence(distribution)
        
        # PHASE 2: Taxonomy Classification
        bias_diagnosis = self.classifier.classify(raw_response, jsd_score, distribution)
        
        # PHASE 3: Mitigation Routing
        routing_decision = self.router.route(bias_diagnosis, original_prompt, raw_response)
        
        final_response = raw_response
        mitigation_applied = routing_decision["action"]
        
        # If bias is detected (JSD > Threshold), regenerate with Neutrality Envelope
        if routing_decision["requires_regeneration"]:
            revised_prompt = routing_decision["revised_prompt"]
            final_response = self.target_llm.generate(revised_prompt)
            
        return {
            "prompt": original_prompt,
            "raw_baseline_response": raw_response,
            "initial_distribution": distribution,
            "initial_jsd_score": jsd_score,
            "diagnosis": bias_diagnosis.value,
            "mitigation_applied": mitigation_applied,
            "final_mitigated_response": final_response
        }
