import os

file_path = "qwen2_5_coder_1_5b_eval_pipeline.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix literal newlines in print statements that cause syntax errors
# The problematic text is literally:
# print("
# " + "="*60)
broken_print = "print(\"\n\" + \"=\"*60)"
fixed_print = "print(\"\\n\" + \"=\"*60)"
content = content.replace(broken_print, fixed_print)

# The user already manually fixed Cell 6. It looks like:
# print("" + "="*60)
# We can fix that to match the others if needed, but it's fine.

# Write the experimental cell to the end if not already present
experimental_cell = """

# %%
# Cell 13: Experimental INT8 HumanEval Batched Generation
import os
os.environ["HUMAN_EVAL_ALLOW_EXECUTION"] = "1"
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from human_eval.evaluation import evaluate_functional_correctness

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"

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

print("\\n" + "="*60)
print("CELL 13: Experimental INT8 HumanEval Batched Generation")
print("="*60)

clear_vram()

print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Loading HumanEval Dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 4

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Generating HumanEval Batched"):
    batch = dataset[i:i + BATCH_SIZE]
    prompts = batch["prompt"]
    task_ids = batch["task_id"]
    
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
        
        stop_words = ["\\ndef ", "\\nclass ", "\\nif __name__ == ", "\\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "/content/qwen_humaneval_int8_batched_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
os.makedirs("/content", exist_ok=True)
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\\n")

print("[*] Running Official HumanEval Grader...")
results = evaluate_functional_correctness(
    sample_file=samples_file,
    k=[1],
    n_workers=4,
    timeout=3.0
)
pass_at_1 = results.get("pass@1", 0.0)

s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

print("\\n--- INT8 HumanEval Batched Results ---")
print(f"Total Samples: {len(dataset)}")
print(f"Batch Size:    {BATCH_SIZE}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Pass@1:        {pass_at_1:.4f}")
print(f"Latency:       {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int8
clear_vram()
"""

if "Cell 13: Experimental INT8 HumanEval Batched Generation" not in content:
    content += experimental_cell

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixes and appending complete.")
