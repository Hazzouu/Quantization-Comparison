# %% [markdown]
# # Llama-3.2-1B-Instruct Quantization Evaluation Pipeline (A100)
# ZERO-DEPENDENCY ARCHITECTURE: Every execution cell is 100% standalone.

# %%
# Cell 1: Package Installations
# !pip install transformers datasets accelerate bitsandbytes pynvml torch lm-eval
    
# %%
# Cell 2: FP16 MMLU & Telemetry (lm-eval)
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

pynvml.nvmlInit()
try:
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"Failed to get NVML handle: {e}")
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

def save_dataset_logits(model, tokenizer, filename, device="cuda"):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_mmlu_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL MMLU evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["mmlu"],
        num_fewshot=5,
        log_samples=False
    )
    acc = results["results"]["mmlu"].get("acc,none", 0.0)
    return acc

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 2: FP16 MMLU Evaluation")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

fp16_acc = run_mmlu_eval(model_fp16, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_fp16, tokenizer, "fp16_logits.pt")

new_metrics = {
    "FP16": {
        "MMLU Acc": fp16_acc,
        "Time (s)": elapsed_time,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
}
update_metrics(new_metrics)

print("\n--- FP16 MMLU Results ---")
print(f"Accuracy:  {fp16_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 3: FP16 Perplexity
import os
import gc
import time
import json
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()
    print("[*] VRAM cleared successfully.")

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    print("[*] Loading wikitext-2-raw-v1 test split for Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = min(model.config.max_position_embeddings, 4096)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    
    print(f"[*] Calculating PPL (Seq Len: {seq_len}, Stride: {stride}, Max Len: {max_length})...")
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 3: FP16 Perplexity")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("[*] Reloading Model in FP16 for PPL...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

fp16_ppl = calculate_perplexity(model_fp16, tokenizer)
update_metrics({"FP16": {"PPL": fp16_ppl}})

print(f"\n--- FP16 PPL Result ---")
print(f"Perplexity: {fp16_ppl:.4f}")

del model_fp16
clear_vram()

# %%
# Cell 4: INT8 Dynamic MMLU & Telemetry
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval
from lm_eval.models.huggingface import HFLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

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

def save_dataset_logits(model, tokenizer, filename, device="cuda"):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_mmlu_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL MMLU evaluation...")
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=["mmlu"], num_fewshot=5, log_samples=False)
    return results["results"]["mmlu"].get("acc,none", 0.0)

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 4: INT8 Dynamic MMLU Evaluation")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading Model in 8-bit...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

int8_acc = run_mmlu_eval(model_int8, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_int8, tokenizer, "int8_logits.pt")

update_metrics({
    "INT8": {
        "MMLU Acc": int8_acc,
        "Time (s)": elapsed_time,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 MMLU Results ---")
print(f"Accuracy:  {int8_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 5: INT8 Dynamic Perplexity
import os
import gc
import time
import json
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()
    print("[*] VRAM cleared successfully.")

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    print("[*] Loading wikitext-2-raw-v1 test split for Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = min(model.config.max_position_embeddings, 4096)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    
    print(f"[*] Calculating PPL (Seq Len: {seq_len}, Stride: {stride}, Max Len: {max_length})...")
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 5: INT8 Dynamic Perplexity")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("[*] Reloading Model in 8-bit for PPL...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

int8_ppl = calculate_perplexity(model_int8, tokenizer)
update_metrics({"INT8": {"PPL": int8_ppl}})

print(f"\n--- INT8 PPL Result ---")
print(f"Perplexity: {int8_ppl:.4f}")

del model_int8
clear_vram()

# %%
# Cell 6: INT4 Dynamic (NF4) MMLU & Telemetry
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval
from lm_eval.models.huggingface import HFLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

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

def save_dataset_logits(model, tokenizer, filename, device="cuda"):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_mmlu_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL MMLU evaluation...")
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=["mmlu"], num_fewshot=5, log_samples=False)
    return results["results"]["mmlu"].get("acc,none", 0.0)

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 6: INT4 NF4 MMLU Evaluation")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading Model in 4-bit NF4...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

int4_acc = run_mmlu_eval(model_int4, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_int4, tokenizer, "int4_logits.pt")

update_metrics({
    "INT4": {
        "MMLU Acc": int4_acc,
        "Time (s)": elapsed_time,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 MMLU Results ---")
print(f"Accuracy:  {int4_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_int4
clear_vram()

# %%
# Cell 7: INT4 Dynamic (NF4) Perplexity
import os
import gc
import time
import json
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    torch.cuda.reset_peak_memory_stats()
    print("[*] VRAM cleared successfully.")

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    print("[*] Loading wikitext-2-raw-v1 test split for Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = min(model.config.max_position_embeddings, 4096)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    
    print(f"[*] Calculating PPL (Seq Len: {seq_len}, Stride: {stride}, Max Len: {max_length})...")
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("pipeline_metrics.json"):
        try:
            with open("pipeline_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 7: INT4 NF4 Perplexity")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("[*] Reloading Model in 4-bit NF4 for PPL...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

int4_ppl = calculate_perplexity(model_int4, tokenizer)
update_metrics({"INT4": {"PPL": int4_ppl}})

print(f"\n--- INT4 PPL Result ---")
print(f"Perplexity: {int4_ppl:.4f}")

del model_int4
clear_vram()

# %%
# Cell 8: KL Divergence & Final Report
import os
import json
import torch
import torch.nn.functional as F

print("\n" + "="*60)
print("CELL 8: KL Divergence & Final Report")
print("="*60)

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("fp16_logits.pt")
int8_logits = torch.load("int8_logits.pt")
int4_logits = torch.load("int4_logits.pt")

def compute_kl(logits_p, logits_q):
    """
    Computes exact KL Divergence KL(P || Q) where P is the reference FP16 distribution
    and Q is the quantized distribution.
    """
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

print("[*] Computing KL Divergence shifts...")
kl_fp16_int8 = compute_kl(fp16_logits, int8_logits)
kl_fp16_int4 = compute_kl(fp16_logits, int4_logits)

with open("pipeline_metrics.json", "r") as f:
    final_metrics = json.load(f)
    
final_metrics["FP16"]["KL Divergence"] = 0.0
final_metrics["INT8"]["KL Divergence"] = kl_fp16_int8
final_metrics["INT4"]["KL Divergence"] = kl_fp16_int4

print("\n" + "="*95)
print(f"{'FINAL HEALTH CARD: QUANTIZATION PROFILING':^95}")
print("="*95)
print(f"{'Metric':<20} | {'FP16 (Baseline)':<20} | {'INT8 Dynamic':<22} | {'INT4 (NF4) Dynamic':<22}")
print("-" * 95)

metrics_keys = ["MMLU Acc", "PPL", "Time (s)", "Avg Power (W)", "Peak VRAM (GB)", "KL Divergence"]

for key in metrics_keys:
    v1 = final_metrics.get("FP16", {}).get(key, 'N/A')
    v2 = final_metrics.get("INT8", {}).get(key, 'N/A')
    v3 = final_metrics.get("INT4", {}).get(key, 'N/A')
    
    v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
    v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
    v3_str = f"{v3:.4f}" if isinstance(v3, float) else str(v3)
    
    print(f"{key:<20} | {v1_str:<20} | {v2_str:<22} | {v3_str:<22}")

print("="*95)
print("[*] Pipeline execution finished successfully.")
