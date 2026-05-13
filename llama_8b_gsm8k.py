# %%
# Cell 1: Environment Setup
# !pip install transformers datasets evaluate accelerate bitsandbytes pynvml torch scikit-learn lm-eval

# %%
# Cell 2: FP16 Baseline Evaluation
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass
    print("[*] VRAM cleared successfully.")

def measure_generative_latency(model, tokenizer, device="cuda"):
    prompt = "Explain the concept of quantum computing in detail:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    end_time = time.time()
    
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    latency = end_time - start_time
    tps = gen_tokens / latency
    return tps

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

def save_kl_logits(model, tokenizer, filename, device="cuda"):
    print("[*] Saving KL logits on 100 sequences of WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename}")

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper for GSM8K...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL GSM8K evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["gsm8k"],
        num_fewshot=5,
        log_samples=False
    )
    exact_match = results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)
    return exact_match

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("gsm8k_metrics.json"):
        try:
            with open("gsm8k_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("gsm8k_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 2: FP16 Baseline Evaluation")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Starting Hardware Telemetry & Task Evaluation...")
tracker = PowerTracker()
tracker.start()

fp16_acc = run_gsm8k_eval(model_fp16, tokenizer)
fp16_ppl = calculate_perplexity(model_fp16, tokenizer)
fp16_tps = measure_generative_latency(model_fp16, tokenizer)

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_kl_logits(model_fp16, tokenizer, "gsm8k_fp16_logits.pt")

new_metrics = {
    "FP16": {
        "GSM8K Exact Match": fp16_acc,
        "PPL": fp16_ppl,
        "Tokens/sec": fp16_tps,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
}
update_metrics(new_metrics)

print("\n--- FP16 Results ---")
print(f"GSM8K Exact Match: {fp16_acc:.4f}")
print(f"Perplexity:        {fp16_ppl:.4f}")
print(f"Gen Latency:       {fp16_tps:.2f} t/s")
print(f"Avg Power:         {avg_power:.2f} W")
print(f"Peak VRAM:         {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 3: INT8 Dynamic Quantization
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass
    print("[*] VRAM cleared successfully.")

def measure_generative_latency(model, tokenizer, device="cuda"):
    prompt = "Explain the concept of quantum computing in detail:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    end_time = time.time()
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    latency = end_time - start_time
    tps = gen_tokens / latency
    return tps

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = min(model.config.max_position_embeddings, 4096)
    seq_len = encodings.input_ids.size(1)
    nlls = []
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

def save_kl_logits(model, tokenizer, filename, device="cuda"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=["gsm8k"], num_fewshot=5, log_samples=False)
    return results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("gsm8k_metrics.json"):
        try:
            with open("gsm8k_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("gsm8k_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 3: INT8 Dynamic Quantization")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Task Evaluation...")
tracker = PowerTracker()
tracker.start()

int8_acc = run_gsm8k_eval(model_int8, tokenizer)
int8_ppl = calculate_perplexity(model_int8, tokenizer)
int8_tps = measure_generative_latency(model_int8, tokenizer)

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_kl_logits(model_int8, tokenizer, "gsm8k_int8_logits.pt")

new_metrics = {
    "INT8": {
        "GSM8K Exact Match": int8_acc,
        "PPL": int8_ppl,
        "Tokens/sec": int8_tps,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
}
update_metrics(new_metrics)

print("\n--- INT8 Results ---")
print(f"GSM8K Exact Match: {int8_acc:.4f}")
print(f"Perplexity:        {int8_ppl:.4f}")
print(f"Gen Latency:       {int8_tps:.2f} t/s")
print(f"Avg Power:         {avg_power:.2f} W")
print(f"Peak VRAM:         {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 4: INT4 NF4 Quantization
import os
import gc
import time
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError:
        pass
    print("[*] VRAM cleared successfully.")

def measure_generative_latency(model, tokenizer, device="cuda"):
    prompt = "Explain the concept of quantum computing in detail:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    end_time = time.time()
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    latency = end_time - start_time
    tps = gen_tokens / latency
    return tps

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    max_length = min(model.config.max_position_embeddings, 4096)
    seq_len = encodings.input_ids.size(1)
    nlls = []
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

def save_kl_logits(model, tokenizer, filename, device="cuda"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=["gsm8k"], num_fewshot=5, log_samples=False)
    return results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("gsm8k_metrics.json"):
        try:
            with open("gsm8k_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("gsm8k_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 4: INT4 NF4 Quantization")
print("="*60)

clear_vram()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT4 NF4 Model...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Task Evaluation...")
tracker = PowerTracker()
tracker.start()

int4_acc = run_gsm8k_eval(model_int4, tokenizer)
int4_ppl = calculate_perplexity(model_int4, tokenizer)
int4_tps = measure_generative_latency(model_int4, tokenizer)

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_kl_logits(model_int4, tokenizer, "gsm8k_int4_logits.pt")

new_metrics = {
    "INT4": {
        "GSM8K Exact Match": int4_acc,
        "PPL": int4_ppl,
        "Tokens/sec": int4_tps,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
}
update_metrics(new_metrics)

print("\n--- INT4 Results ---")
print(f"GSM8K Exact Match: {int4_acc:.4f}")
print(f"Perplexity:        {int4_ppl:.4f}")
print(f"Gen Latency:       {int4_tps:.2f} t/s")
print(f"Avg Power:         {avg_power:.2f} W")
print(f"Peak VRAM:         {peak_vram:.2f} GB")

del model_int4
clear_vram()

# %%
# Cell 5: KL Divergence Calculation
import os
import json
import torch
import torch.nn.functional as F

print("\n" + "="*60)
print("CELL 5: KL Divergence Calculation (No Model Loaded)")
print("="*60)

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("gsm8k_fp16_logits.pt")
int8_logits = torch.load("gsm8k_int8_logits.pt")
int4_logits = torch.load("gsm8k_int4_logits.pt")

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

with open("gsm8k_metrics.json", "r") as f:
    final_metrics = json.load(f)
    
final_metrics["FP16"]["KL Divergence"] = 0.0
final_metrics["INT8"]["KL Divergence"] = kl_fp16_int8
final_metrics["INT4"]["KL Divergence"] = kl_fp16_int4

print("\n" + "="*115)
print(f"{'FINAL HEALTH CARD: QUANTIZATION PROFILING (GSM8K)':^115}")
print("="*115)
print(f"{'Metric':<25} | {'FP16 (Baseline)':<25} | {'INT8 Dynamic':<25} | {'INT4 (NF4) Dynamic':<25}")
print("-" * 115)

metrics_keys = ["GSM8K Exact Match", "PPL", "Tokens/sec", "Avg Power (W)", "Peak VRAM (GB)", "KL Divergence"]

for key in metrics_keys:
    v1 = final_metrics.get("FP16", {}).get(key, 'N/A')
    v2 = final_metrics.get("INT8", {}).get(key, 'N/A')
    v3 = final_metrics.get("INT4", {}).get(key, 'N/A')
    
    v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
    v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
    v3_str = f"{v3:.4f}" if isinstance(v3, float) else str(v3)
    
    print(f"{key:<25} | {v1_str:<25} | {v2_str:<25} | {v3_str:<25}")

print("="*115)
print("[*] GSM8K Pipeline execution finished successfully.")
