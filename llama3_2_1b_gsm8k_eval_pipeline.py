# %% [markdown]
# # Llama-3.2-1B-Instruct GSM8K Quantization Evaluation (A100)
# ZERO-DEPENDENCY ARCHITECTURE: Every execution cell is 100% standalone.
# Evaluates GSM8K Accuracy across FP16, INT8, and INT4 (NF4).

# %%
# Cell 1: Package Installations
# !pip install transformers datasets accelerate bitsandbytes pynvml torch lm-eval

# %%
# Cell 2: FP16 GSM8K & Telemetry
import os
import gc
import time
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

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL GSM8K evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["gsm8k"],
        num_fewshot=5,
        log_samples=False
    )
    acc = results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)
    return acc

print("\n" + "="*60)
print("CELL 2: FP16 GSM8K Evaluation")
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

fp16_acc = run_gsm8k_eval(model_fp16, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\n--- FP16 GSM8K Results ---")
print(f"Accuracy:  {fp16_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 3: INT8 Dynamic GSM8K & Telemetry
import os
import gc
import time
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

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL GSM8K evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["gsm8k"],
        num_fewshot=5,
        log_samples=False
    )
    acc = results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)
    return acc

print("\n" + "="*60)
print("CELL 3: INT8 Dynamic GSM8K Evaluation")
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

int8_acc = run_gsm8k_eval(model_int8, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\n--- INT8 GSM8K Results ---")
print(f"Accuracy:  {int8_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 4: INT4 Dynamic (NF4) GSM8K & Telemetry
import os
import gc
import time
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

def run_gsm8k_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper...")
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print("[*] Running FULL GSM8K evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["gsm8k"],
        num_fewshot=5,
        log_samples=False
    )
    acc = results["results"]["gsm8k"].get("exact_match,strict-match", 0.0)
    return acc

print("\n" + "="*60)
print("CELL 4: INT4 NF4 GSM8K Evaluation")
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

int4_acc = run_gsm8k_eval(model_int4, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\n--- INT4 GSM8K Results ---")
print(f"Accuracy:  {int4_acc:.4f}")
print(f"Time Est:  {elapsed_time:.2f} seconds")
print(f"Avg Power: {avg_power:.2f} W")
print(f"Peak VRAM: {peak_vram:.2f} GB")

del model_int4
clear_vram()
