# %% [markdown]
# # Qwen2.5-Coder-1.5B Batched MBPP Pipeline
# Offline Evaluation with Batched Generation for A100

# %%
# Cell 1: Package Installations
"!pip install transformers datasets accelerate bitsandbytes pynvml torch evalplus tqdm"

# %%
# Cell 3: FP16 MBPP Batched Generation
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
from tqdm import tqdm

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

print("\n" + "="*60)
print("CELL 3: FP16 MBPP Batched Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Loading MBPP Dataset...")
mbpp_data = list(get_mbpp_plus().items())

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 32

for i in tqdm(range(0, len(mbpp_data), BATCH_SIZE), desc="Batched MBPP Generation"):
    batch = mbpp_data[i:i + BATCH_SIZE]
    task_ids = [item[0] for item in batch]
    prompts = [item[1]["prompt"] for item in batch]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model_fp16.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    for j, output in enumerate(generated_ids):
        new_tokens = output[input_length:]
        total_generated_tokens += len(new_tokens)
        
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        stop_words = ["\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "solution": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_mbpp_fp16_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Sanitizing model outputs...")
# 1. Run the sanitizer to strip markdown and bad asserts
os.system(f"evalplus.sanitize --samples {samples_file}")

sanitized_file = samples_file.replace(".jsonl", "-sanitized.jsonl")

print("[*] Running Official EvalPlus MBPP Grader...")
# 2. Grade the CLEANED file
os.system(f"evalplus.evaluate --dataset mbpp --samples {sanitized_file} --i-just-wanna-run")

# 3. Update the read path to point to the new sanitized results
import glob
result_files = glob.glob("*_eval_results.json")
base_pass_1 = 0.0
plus_pass_1 = 0.0
if result_files:
    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        data = json.load(f)
    eval_data = data.get("eval", {})
    total_tasks = len(eval_data)
    
    base_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("base_status") == "pass")
    plus_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("plus_status") == "pass")
    
    if total_tasks > 0:
        base_pass_1 = base_passes / total_tasks
        plus_pass_1 = plus_passes / total_tasks

s_per_it = elapsed_time / len(mbpp_data)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "FP16": {
        "MBPP Base Pass@1": base_pass_1,
        "MBPP+ Pass@1": plus_pass_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 MBPP Batched Results ---")
print(f"Total Samples: {len(mbpp_data)}")
print(f"Batch Size:    {BATCH_SIZE}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Base Pass@1:   {base_pass_1:.4f}")
print(f"MBPP+ Pass@1:  {plus_pass_1:.4f}")
print(f"Latency:       {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 6: INT8 MBPP Batched Generation
print("\n" + "="*60)
print("CELL 6: INT8 Dynamic MBPP Batched Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 32

for i in tqdm(range(0, len(mbpp_data), BATCH_SIZE), desc="Batched MBPP Generation"):
    batch = mbpp_data[i:i + BATCH_SIZE]
    task_ids = [item[0] for item in batch]
    prompts = [item[1]["prompt"] for item in batch]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model_int8.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    for j, output in enumerate(generated_ids):
        new_tokens = output[input_length:]
        total_generated_tokens += len(new_tokens)
        
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        stop_words = ["\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "solution": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_mbpp_int8_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Sanitizing model outputs...")
# 1. Run the sanitizer to strip markdown and bad asserts
os.system(f"evalplus.sanitize --samples {samples_file}")

sanitized_file = samples_file.replace(".jsonl", "-sanitized.jsonl")

print("[*] Running Official EvalPlus MBPP Grader...")
# 2. Grade the CLEANED file
os.system(f"evalplus.evaluate --dataset mbpp --samples {sanitized_file} --i-just-wanna-run")

# 3. Update the read path to point to the new sanitized results
import glob
result_files = glob.glob("*_eval_results.json")
base_pass_1 = 0.0
plus_pass_1 = 0.0
if result_files:
    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        data = json.load(f)
    eval_data = data.get("eval", {})
    total_tasks = len(eval_data)
    
    base_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("base_status") == "pass")
    plus_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("plus_status") == "pass")
    
    if total_tasks > 0:
        base_pass_1 = base_passes / total_tasks
        plus_pass_1 = plus_passes / total_tasks

s_per_it = elapsed_time / len(mbpp_data)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT8": {
        "MBPP Base Pass@1": base_pass_1,
        "MBPP+ Pass@1": plus_pass_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 MBPP Batched Results ---")
print(f"Total Samples: {len(mbpp_data)}")
print(f"Batch Size:    {BATCH_SIZE}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Base Pass@1:   {base_pass_1:.4f}")
print(f"MBPP+ Pass@1:  {plus_pass_1:.4f}")
print(f"Latency:       {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 9: INT4 MBPP Batched Generation
print("\n" + "="*60)
print("CELL 9: INT4 Dynamic MBPP Batched Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT4 Model...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 32

for i in tqdm(range(0, len(mbpp_data), BATCH_SIZE), desc="Batched MBPP Generation"):
    batch = mbpp_data[i:i + BATCH_SIZE]
    task_ids = [item[0] for item in batch]
    prompts = [item[1]["prompt"] for item in batch]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model_int4.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    for j, output in enumerate(generated_ids):
        new_tokens = output[input_length:]
        total_generated_tokens += len(new_tokens)
        
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        stop_words = ["\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "solution": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_mbpp_int4_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Sanitizing model outputs...")
# 1. Run the sanitizer to strip markdown and bad asserts
os.system(f"evalplus.sanitize --samples {samples_file}")

sanitized_file = samples_file.replace(".jsonl", "-sanitized.jsonl")

print("[*] Running Official EvalPlus MBPP Grader...")
# 2. Grade the CLEANED file
os.system(f"evalplus.evaluate --dataset mbpp --samples {sanitized_file} --i-just-wanna-run")

# 3. Update the read path to point to the new sanitized results
import glob
result_files = glob.glob("*_eval_results.json")
base_pass_1 = 0.0
plus_pass_1 = 0.0
if result_files:
    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        data = json.load(f)
    eval_data = data.get("eval", {})
    total_tasks = len(eval_data)
    
    base_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("base_status") == "pass")
    plus_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("plus_status") == "pass")
    
    if total_tasks > 0:
        base_pass_1 = base_passes / total_tasks
        plus_pass_1 = plus_passes / total_tasks

s_per_it = elapsed_time / len(mbpp_data)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT4": {
        "MBPP Base Pass@1": base_pass_1,
        "MBPP+ Pass@1": plus_pass_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 MBPP Batched Results ---")
print(f"Total Samples: {len(mbpp_data)}")
print(f"Batch Size:    {BATCH_SIZE}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Base Pass@1:   {base_pass_1:.4f}")
print(f"MBPP+ Pass@1:  {plus_pass_1:.4f}")
print(f"Latency:       {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int4
clear_vram()
