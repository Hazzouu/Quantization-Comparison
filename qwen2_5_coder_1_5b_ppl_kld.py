# %% [markdown]
# # Qwen2.5-Coder-1.5B Perplexity (Wikitext) Pipeline
# KL Divergence & Perplexity Profiling

# %%
# Cell 1: Package Installations
import os
os.system('pip install transformers datasets accelerate bitsandbytes pynvml torch lm-eval tqdm')

# %%
# Cell 4: FP16 Perplexity & Logits
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evalplus.data import get_mbpp_plus
import torch.nn.functional as F
import lm_eval
from lm_eval.models.huggingface import HFLM

pynvml.nvmlInit()
try:
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    nvml_handle = None

class PowerTracker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.power_readings = []

    def run(self):
        while not self.stop_event.is_set():
            if nvml_handle:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                self.power_readings.append(power_mw / 1000.0)
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()
    print("[*] VRAM cleared successfully.")

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("qwen_pipeline_metrics.json"):
        try:
            with open("qwen_pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("qwen_pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

def run_ppl_eval(model, tokenizer):
    TASK = "wikitext"
    print(f"[*] Initializing lm-eval HFLM wrapper for {TASK}...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer)
    
    print(f"[*] Running {TASK} evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[TASK],
        num_fewshot=0,
        batch_size=1
    )
    
    ppl = results["results"][TASK].get("word_perplexity,none", results["results"][TASK].get("word_perplexity", 0.0))
    return ppl

print("\n" + "="*60)
print("CELL 4: FP16 Perplexity Evaluation (Wikitext)")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Starting Hardware Telemetry & Evaluation...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_fp16, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

update_metrics({
    "FP16": {
        "PPL (Wikitext)": ppl_score,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 Perplexity Results ---")
print(f"Dataset:       Wikitext")
print(f"Perplexity:    {ppl_score:.4f}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

# Compute base logits for KL divergence on a small fixed text
print("[*] Computing FP16 Reference Logits for KL Divergence (MBPP Subset)...")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
prompts = [task["prompt"] for task_id, task in list(get_mbpp_plus().items())[:16]]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    outputs = model_fp16(**inputs)
    fp16_logits = outputs.logits.detach().cpu()
    torch.save(fp16_logits, "fp16_reference_logits.pt")
print("[*] FP16 reference logits saved to fp16_reference_logits.pt")

del model_fp16
clear_vram()

# %%
# Cell 7: INT8 Perplexity & KL Divergence
print("\n" + "="*60)
print("CELL 7: INT8 Dynamic Perplexity & KL Divergence")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Evaluation...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_int8, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("[*] Computing INT8 Logits for KL Divergence (MBPP Subset)...")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
from evalplus.data import get_mbpp_plus
prompts = [task["prompt"] for task_id, task in list(get_mbpp_plus().items())[:16]]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    outputs = model_int8(**inputs)
    int8_logits = outputs.logits.detach().cpu()

kl_div = 0.0
if os.path.exists("fp16_reference_logits.pt"):
    fp16_logits = torch.load("fp16_reference_logits.pt")
    
    import torch.nn.functional as F
    
    # Flatten [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
    # This guarantees we get the true Average KL per token
    flat_int8_logits = int8_logits.view(-1, int8_logits.size(-1))
    flat_fp16_logits = fp16_logits.view(-1, fp16_logits.size(-1))

    q_log_probs = F.log_softmax(flat_int8_logits, dim=-1)
    p_probs = F.softmax(flat_fp16_logits, dim=-1)
    
    kl_div_tensor = F.kl_div(q_log_probs, p_probs, reduction='batchmean')
    kl_div = kl_div_tensor.item()
    print(f"[*] Computed KL Divergence vs FP16: {kl_div:.6f}")
else:
    print("[!] Warning: fp16_reference_logits.pt not found. Cannot compute KL divergence.")

update_metrics({
    "INT8": {
        "PPL (Wikitext)": ppl_score,
        "KL Divergence": kl_div,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 Perplexity Results ---")
print(f"Dataset:       Wikitext")
print(f"Perplexity:    {ppl_score:.4f}")
print(f"KL Divergence: {kl_div:.6f}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 10: INT4 Perplexity & KL Divergence
print("\n" + "="*60)
print("CELL 10: INT4 Dynamic Perplexity & KL Divergence")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("[*] Loading INT4 Model...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Evaluation...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_int4, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("[*] Computing INT4 Logits for KL Divergence (MBPP Subset)...")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
from evalplus.data import get_mbpp_plus
prompts = [task["prompt"] for task_id, task in list(get_mbpp_plus().items())[:16]]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

with torch.no_grad():
    outputs = model_int4(**inputs)
    int4_logits = outputs.logits.detach().cpu()

kl_div = 0.0
if os.path.exists("fp16_reference_logits.pt"):
    fp16_logits = torch.load("fp16_reference_logits.pt")
    
    import torch.nn.functional as F
    
    # Flatten [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
    # This guarantees we get the true Average KL per token
    flat_int4_logits = int4_logits.view(-1, int4_logits.size(-1))
    flat_fp16_logits = fp16_logits.view(-1, fp16_logits.size(-1))

    q_log_probs = F.log_softmax(flat_int4_logits, dim=-1)
    p_probs = F.softmax(flat_fp16_logits, dim=-1)
    
    kl_div_tensor = F.kl_div(q_log_probs, p_probs, reduction='batchmean')
    kl_div = kl_div_tensor.item()
    print(f"[*] Computed KL Divergence vs FP16: {kl_div:.6f}")
else:
    print("[!] Warning: fp16_reference_logits.pt not found. Cannot compute KL divergence.")

update_metrics({
    "INT4": {
        "PPL (Wikitext)": ppl_score,
        "KL Divergence": kl_div,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 Perplexity Results ---")
print(f"Dataset:       Wikitext")
print(f"Perplexity:    {ppl_score:.4f}")
print(f"KL Divergence: {kl_div:.6f}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int4
clear_vram()
