# %% [markdown]
# # Qwen2.5-Coder-7B HumanEval Quantization Pipeline
# Offline Evaluation with Batched Generation for A100

# %%
# Cell 1: Package Installations
!pip install transformers datasets evaluate accelerate bitsandbytes pynvml torch scikit-learn evalplus git+https://github.com/openai/human-eval.git

# %%
# Cell 2: FP16 Baseline Evaluation
import os
os.environ["HUMAN_EVAL_ALLOW_EXECUTION"] = "1"
os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O /content/HumanEval.jsonl.gz')

import gc
import time
import json
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from human_eval.evaluation import evaluate_functional_correctness

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
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    print("[*] VRAM cleared successfully.")

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("qwen_7b_humaneval_metrics.json"):
        try:
            with open("qwen_7b_humaneval_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("qwen_7b_humaneval_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 2: FP16 HumanEval Batched Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Loading HumanEval Dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 32

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="HumanEval FP16 Generation"):
    batch = dataset[i:i + BATCH_SIZE]
    prompts = batch["prompt"]
    task_ids = batch["task_id"]
    
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
        
        stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_7b_humaneval_fp16_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official HumanEval Grader...")
results = evaluate_functional_correctness(
    sample_file=samples_file,
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results.get("pass@1", 0.0)
s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

print("[*] Running WikiText-2 PPL Evaluation...")
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
wiki_encodings = tokenizer("\n\n".join(wiki_data["text"]), return_tensors="pt")
max_length = min(model_fp16.config.max_position_embeddings, 4096)
stride = 512
seq_len = wiki_encodings.input_ids.size(1)
nlls = []
for i in tqdm(range(0, seq_len, stride), desc="PPL Calculation"):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, seq_len)
    trg_len = end_loc - i
    input_ids = wiki_encodings.input_ids[:, begin_loc:end_loc].to("cuda")
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100 
    with torch.no_grad():
        outputs = model_fp16(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
ppl_val = torch.exp(torch.stack(nlls).mean()).item()

print("[*] Extracting HumanEval Logits (100 samples) for KL...")
kl_dataset = dataset.select(range(100))
# Force conversion to a pristine Python list of strings to bypass pyarrow conflicts
kl_prompts = [str(p) for p in kl_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    kl_outputs = model_fp16(**kl_inputs)
torch.save(kl_outputs.logits.cpu(), "qwen_7b_humaneval_fp16_logits.pt")

update_metrics({
    "FP16": {
        "HumanEval Pass@1": pass_at_1,
        "WikiText-2 PPL": ppl_val,
        "Time (s)": elapsed_time,
        "Latency s/it": s_per_it,
        "Generative Latency t/s": t_per_s,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 Results ---")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"PPL:        {ppl_val:.4f}")
print(f"Time:       {elapsed_time:.2f} s")
print(f"Tokens/Sec: {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

for var in ['inputs', 'generated_ids', 'wiki_encodings', 'input_ids', 'target_ids', 'outputs', 'kl_prompts', 'kl_inputs', 'kl_outputs', 'nlls', 'batch']:
    if var in globals():
        del globals()[var]
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
from datasets import load_dataset
from tqdm import tqdm
from human_eval.evaluation import evaluate_functional_correctness

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
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    print("[*] VRAM cleared successfully.")

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("qwen_7b_humaneval_metrics.json"):
        try:
            with open("qwen_7b_humaneval_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("qwen_7b_humaneval_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 3: INT8 Dynamic Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
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
BATCH_SIZE = 32

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="HumanEval INT8 Generation"):
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
        
        stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_7b_humaneval_int8_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official HumanEval Grader...")
results = evaluate_functional_correctness(
    sample_file=samples_file,
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results.get("pass@1", 0.0)
s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

print("[*] Running WikiText-2 PPL Evaluation...")
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
wiki_encodings = tokenizer("\n\n".join(wiki_data["text"]), return_tensors="pt")
max_length = min(model_int8.config.max_position_embeddings, 4096)
stride = 512
seq_len = wiki_encodings.input_ids.size(1)
nlls = []
for i in tqdm(range(0, seq_len, stride), desc="PPL Calculation"):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, seq_len)
    trg_len = end_loc - i
    input_ids = wiki_encodings.input_ids[:, begin_loc:end_loc].to("cuda")
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100 
    with torch.no_grad():
        outputs = model_int8(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
ppl_val = torch.exp(torch.stack(nlls).mean()).item()

print("[*] Extracting HumanEval Logits (100 samples) for KL...")
kl_dataset = dataset.select(range(100))
# Force conversion to a pristine Python list of strings to bypass pyarrow conflicts
kl_prompts = [str(p) for p in kl_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    kl_outputs = model_int8(**kl_inputs)
torch.save(kl_outputs.logits.cpu(), "qwen_7b_humaneval_int8_logits.pt")

update_metrics({
    "INT8": {
        "HumanEval Pass@1": pass_at_1,
        "WikiText-2 PPL": ppl_val,
        "Time (s)": elapsed_time,
        "Latency s/it": s_per_it,
        "Generative Latency t/s": t_per_s,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 Results ---")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"PPL:        {ppl_val:.4f}")
print(f"Time:       {elapsed_time:.2f} s")
print(f"Tokens/Sec: {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

for var in ['inputs', 'generated_ids', 'wiki_encodings', 'input_ids', 'target_ids', 'outputs', 'kl_prompts', 'kl_inputs', 'kl_outputs', 'nlls', 'batch']:
    if var in globals():
        del globals()[var]
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
from datasets import load_dataset
from tqdm import tqdm
from human_eval.evaluation import evaluate_functional_correctness

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
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    print("[*] VRAM cleared successfully.")

def update_metrics(new_data):
    metrics = {}
    if os.path.exists("qwen_7b_humaneval_metrics.json"):
        try:
            with open("qwen_7b_humaneval_metrics.json", "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
    for k, v in new_data.items():
        if k not in metrics:
            metrics[k] = v
        else:
            metrics[k].update(v)
    with open("qwen_7b_humaneval_metrics.json", "w") as f:
        json.dump(metrics, f)

print("\n" + "="*60)
print("CELL 4: INT4 NF4 Evaluation")
print("="*60)

clear_vram()

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
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

print("[*] Loading HumanEval Dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Batched Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0
BATCH_SIZE = 32

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="HumanEval INT4 Generation"):
    batch = dataset[i:i + BATCH_SIZE]
    prompts = batch["prompt"]
    task_ids = batch["task_id"]
    
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
        
        stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
        for stop_word in stop_words:
            if stop_word in completion:
                completion = completion.split(stop_word)[0]
                
        samples.append({"task_id": task_ids[j], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "qwen_7b_humaneval_int4_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official HumanEval Grader...")
results = evaluate_functional_correctness(
    sample_file=samples_file,
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results.get("pass@1", 0.0)
s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

print("[*] Running WikiText-2 PPL Evaluation...")
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
wiki_encodings = tokenizer("\n\n".join(wiki_data["text"]), return_tensors="pt")
max_length = min(model_int4.config.max_position_embeddings, 4096)
stride = 512
seq_len = wiki_encodings.input_ids.size(1)
nlls = []
for i in tqdm(range(0, seq_len, stride), desc="PPL Calculation"):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, seq_len)
    trg_len = end_loc - i
    input_ids = wiki_encodings.input_ids[:, begin_loc:end_loc].to("cuda")
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100 
    with torch.no_grad():
        outputs = model_int4(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
ppl_val = torch.exp(torch.stack(nlls).mean()).item()

print("[*] Extracting HumanEval Logits (100 samples) for KL...")
kl_dataset = dataset.select(range(100))
# Force conversion to a pristine Python list of strings to bypass pyarrow conflicts
kl_prompts = [str(p) for p in kl_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    kl_outputs = model_int4(**kl_inputs)
torch.save(kl_outputs.logits.cpu(), "qwen_7b_humaneval_int4_logits.pt")

update_metrics({
    "INT4": {
        "HumanEval Pass@1": pass_at_1,
        "WikiText-2 PPL": ppl_val,
        "Time (s)": elapsed_time,
        "Latency s/it": s_per_it,
        "Generative Latency t/s": t_per_s,
        "Avg Power (W)": avg_power,
        "Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 Results ---")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"PPL:        {ppl_val:.4f}")
print(f"Time:       {elapsed_time:.2f} s")
print(f"Tokens/Sec: {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

for var in ['inputs', 'generated_ids', 'wiki_encodings', 'input_ids', 'target_ids', 'outputs', 'kl_prompts', 'kl_inputs', 'kl_outputs', 'nlls', 'batch']:
    if var in globals():
        del globals()[var]
del model_int4
clear_vram()

# %%
# Cell 5: KL Divergence Calculation
import os
import gc
import json
import torch
import torch.nn.functional as F

print("\n" + "="*60)
print("CELL 5: KL Divergence Calculation")
print("="*60)

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    print("[*] VRAM cleared successfully.")
clear_vram()

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("qwen_7b_humaneval_fp16_logits.pt")
int8_logits = torch.load("qwen_7b_humaneval_int8_logits.pt")
int4_logits = torch.load("qwen_7b_humaneval_int4_logits.pt")

vocab_size = fp16_logits.size(-1)

def compute_kl(logits_p, logits_q, vocab_size):
    """
    Computes exact per-token KL Divergence using flattened tensors.
    """
    flat_p = logits_p.view(-1, vocab_size)
    flat_q = logits_q.view(-1, vocab_size)
    p_probs = F.softmax(flat_p, dim=-1)
    q_log_probs = F.log_softmax(flat_q, dim=-1)
    kl = F.kl_div(q_log_probs, p_probs, reduction='batchmean').item()
    return kl

print("[*] Computing KL Divergence shifts...")
kl_fp16_int8 = compute_kl(fp16_logits, int8_logits, vocab_size)
kl_fp16_int4 = compute_kl(fp16_logits, int4_logits, vocab_size)

with open("qwen_7b_humaneval_metrics.json", "r") as f:
    final_metrics = json.load(f)

final_metrics["FP16"]["KL Divergence"] = 0.0
final_metrics["INT8"]["KL Divergence"] = kl_fp16_int8
final_metrics["INT4"]["KL Divergence"] = kl_fp16_int4

with open("qwen_7b_humaneval_metrics.json", "w") as f:
    json.dump(final_metrics, f)

print("\n" + "="*95)
print(f"{'FINAL HEALTH CARD: QUANTIZATION PROFILING (Qwen2.5-Coder-7B HumanEval)':^95}")
print("="*95)
print(f"{'Metric':<25} | {'FP16 (Baseline)':<20} | {'INT8 Dynamic':<20} | {'INT4 (NF4) Dynamic':<20}")
print("-" * 95)

metrics_keys = [
    "HumanEval Pass@1", 
    "WikiText-2 PPL", 
    "Time (s)", 
    "Generative Latency t/s",
    "Avg Power (W)", 
    "Peak VRAM (GB)", 
    "KL Divergence"
]

for key in metrics_keys:
    v1 = final_metrics.get("FP16", {}).get(key, 'N/A')
    v2 = final_metrics.get("INT8", {}).get(key, 'N/A')
    v3 = final_metrics.get("INT4", {}).get(key, 'N/A')
    
    v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
    v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
    v3_str = f"{v3:.4f}" if isinstance(v3, float) else str(v3)
    
    print(f"{key:<25} | {v1_str:<20} | {v2_str:<20} | {v3_str:<20}")

print("="*95)
print("[*] HumanEval Pipeline execution finished successfully.")
clear_vram()
