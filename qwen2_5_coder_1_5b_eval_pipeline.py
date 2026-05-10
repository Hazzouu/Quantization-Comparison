# %% [markdown]
# # Qwen2.5-Coder-1.5B-Instruct Evaluation Pipeline
# ZERO-DEPENDENCY ARCHITECTURE: Every execution cell is 100% standalone.

# %%
# Cell 1: Package Installations
# os.system('pip install transformers datasets accelerate bitsandbytes pynvml torch lm-eval tqdm git+https://github.com/openai/human-eval.git

# %%
# Cell 2: FP16 HumanEval Generation & Grading
import os
os.environ["HUMAN_EVAL_ALLOW_EXECUTION"] = "1"
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O /content/HumanEval.jsonl.gz')


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
print("CELL 2: FP16 HumanEval Generation & Grading")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Loading openai_humaneval dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0

for task in tqdm(dataset, desc="Generating Code"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_fp16.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice to get only newly generated tokens
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task["task_id"], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("[*] Saving samples to /content/qwen_fp16_samples.jsonl...")
os.makedirs("/content", exist_ok=True)
with open("/content/qwen_fp16_samples.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official OpenAI Grader...")
results = evaluate_functional_correctness(
    "/content/qwen_fp16_samples.jsonl",
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results['pass@1']

s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "FP16": {
        "HumanEval Pass@1": pass_at_1,
        "HumanEval Time (s)": elapsed_time,
        "HumanEval s/it": s_per_it,
        "HumanEval t/s (est)": t_per_s,
        "HumanEval Power (W)": avg_power,
        "HumanEval Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 HumanEval Results ---")
print(f"Total Samples: {len(dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 4: FP16 MBPP & Telemetry
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evalplus.data import get_mbpp
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
print("CELL 3: FP16 MBPP Evaluation")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

mbpp_dataset = get_mbpp()
samples = []
total_generated_tokens = 0

for task_id, task in tqdm(mbpp_dataset.items(), desc="Generating MBPP+"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_fp16.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task_id, "solution": completion})
    
elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "/content/qwen_mbpp_fp16_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
os.makedirs("/content", exist_ok=True)
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running EvalPlus MBPP Grader...")
os.system(f"evalplus.evaluate --dataset mbpp --samples {samples_file} --i-just-wanna-run")

results_file = samples_file.replace(".jsonl", "_eval_results.json")
pass_at_1 = 0.0
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        res_data = json.load(f)
        pass_at_1 = res_data["pass@1"]["mbpp"] if "pass@1" in res_data and "mbpp" in res_data["pass@1"] else res_data.get("pass@1", 0.0)
else:
    print(f"[!] Warning: EvalPlus results file not found at {results_file}")

s_per_it = elapsed_time / len(mbpp_dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "FP16": {
        "MBPP Pass@1": pass_at_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 MBPP Results ---")
print(f"Total Samples: {len(mbpp_dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_fp16
clear_vram()

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
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def save_dataset_logits(model, tokenizer, filename, device="cuda"):
    from datasets import load_dataset
    dataset = load_dataset("openai_humaneval", split="test")
    valid_texts = [text for text in dataset["prompt"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_ppl_eval(model, tokenizer, batch_size="auto"):
    print(f"[*] Initializing lm-eval HFLM wrapper for {TASK}...")
    TASK = "wikitext"
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print(f"[*] Running {TASK} evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[TASK],
        num_fewshot=0,
        log_samples=False
    )
    ppl = results["results"][TASK].get("word_perplexity,none", results["results"][TASK].get("word_perplexity", 0.0))
    return ppl

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
print("CELL 4: FP16 Perplexity & Logits")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_fp16, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_fp16, tokenizer, "qwen_fp16_logits.pt")

update_metrics({
    "FP16": {
        "Perplexity": ppl_score,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- FP16 Perplexity Results ---")
print(f"Perplexity: {ppl_score:.4f}")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_fp16
clear_vram()

# %%
# Cell 5: INT8 Dynamic HumanEval Generation & Grading
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
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O /content/HumanEval.jsonl.gz')


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
print("CELL 5: INT8 Dynamic HumanEval Generation & Grading")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Loading openai_humaneval dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0

for task in tqdm(dataset, desc="Generating Code"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_int8.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice to get only newly generated tokens
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task["task_id"], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("[*] Saving samples to /content/qwen_int8_samples.jsonl...")
os.makedirs("/content", exist_ok=True)
with open("/content/qwen_int8_samples.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official OpenAI Grader...")
results = evaluate_functional_correctness(
    "/content/qwen_int8_samples.jsonl",
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results['pass@1']

s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT8": {
        "HumanEval Pass@1": pass_at_1,
        "HumanEval Time (s)": elapsed_time,
        "HumanEval s/it": s_per_it,
        "HumanEval t/s (est)": t_per_s,
        "HumanEval Power (W)": avg_power,
        "HumanEval Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 HumanEval Results ---")
print(f"Total Samples: {len(dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 8: INT8 Dynamic MBPP & Telemetry
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evalplus.data import get_mbpp
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
print("CELL 6: INT8 Dynamic MBPP Evaluation")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

mbpp_dataset = get_mbpp()
samples = []
total_generated_tokens = 0

for task_id, task in tqdm(mbpp_dataset.items(), desc="Generating MBPP+"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_int8.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task_id, "solution": completion})
    
elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "/content/qwen_mbpp_int8_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
os.makedirs("/content", exist_ok=True)
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running EvalPlus MBPP Grader...")
os.system(f"evalplus.evaluate --dataset mbpp --samples {samples_file} --i-just-wanna-run")

results_file = samples_file.replace(".jsonl", "_eval_results.json")
pass_at_1 = 0.0
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        res_data = json.load(f)
        pass_at_1 = res_data["pass@1"]["mbpp"] if "pass@1" in res_data and "mbpp" in res_data["pass@1"] else res_data.get("pass@1", 0.0)
else:
    print(f"[!] Warning: EvalPlus results file not found at {results_file}")

s_per_it = elapsed_time / len(mbpp_dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT8": {
        "MBPP Pass@1": pass_at_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 MBPP Results ---")
print(f"Total Samples: {len(mbpp_dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 7: INT8 Dynamic Perplexity & Logits
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
    dataset = load_dataset("openai_humaneval", split="test")
    valid_texts = [text for text in dataset["prompt"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_ppl_eval(model, tokenizer, batch_size="auto"):
    print(f"[*] Initializing lm-eval HFLM wrapper for {TASK}...")
    TASK = "wikitext"
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print(f"[*] Running {TASK} evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[TASK],
        num_fewshot=0,
        log_samples=False
    )
    ppl = results["results"][TASK].get("word_perplexity,none", results["results"][TASK].get("word_perplexity", 0.0))
    return ppl

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
print("CELL 7: INT8 Dynamic Perplexity & Logits")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_int8, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_int8, tokenizer, "qwen_int8_logits.pt")

update_metrics({
    "INT8": {
        "Perplexity": ppl_score,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT8 Perplexity Results ---")
print(f"Perplexity: {ppl_score:.4f}")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int8
clear_vram()

# %%
# Cell 8: INT4 Dynamic (NF4) HumanEval Generation & Grading
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
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O /content/HumanEval.jsonl.gz')


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
print("CELL 8: INT4 Dynamic HumanEval Generation & Grading")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

print("[*] Loading openai_humaneval dataset...")
dataset = load_dataset("openai_humaneval", split="test")

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

samples = []
total_generated_tokens = 0

for task in tqdm(dataset, desc="Generating Code"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_int4.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice to get only newly generated tokens
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task["task_id"], "completion": completion})

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("[*] Saving samples to /content/qwen_int4_samples.jsonl...")
os.makedirs("/content", exist_ok=True)
with open("/content/qwen_int4_samples.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running Official OpenAI Grader...")
results = evaluate_functional_correctness(
    "/content/qwen_int4_samples.jsonl",
    k=[1],
    problem_file="/content/HumanEval.jsonl.gz"
)
pass_at_1 = results['pass@1']

s_per_it = elapsed_time / len(dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT4": {
        "HumanEval Pass@1": pass_at_1,
        "HumanEval Time (s)": elapsed_time,
        "HumanEval s/it": s_per_it,
        "HumanEval t/s (est)": t_per_s,
        "HumanEval Power (W)": avg_power,
        "HumanEval Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 HumanEval Results ---")
print(f"Total Samples: {len(dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int4
clear_vram()

# %%
# Cell 12: INT4 Dynamic (NF4) MBPP & Telemetry
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import gc
import time
import json
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evalplus.data import get_mbpp
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
print("CELL 9: INT4 Dynamic MBPP Evaluation")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

print("[*] Starting Hardware Telemetry & Generation Loop...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

mbpp_dataset = get_mbpp()
samples = []
total_generated_tokens = 0

for task_id, task in tqdm(mbpp_dataset.items(), desc="Generating MBPP+"):
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model_int4.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][input_length:]
    total_generated_tokens += len(new_tokens)
    
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
    for stop_word in stop_words:
        if stop_word in completion:
            completion = completion.split(stop_word)[0]
            
    samples.append({"task_id": task_id, "solution": completion})
    
elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

samples_file = "/content/qwen_mbpp_int4_samples.jsonl"
print(f"[*] Saving samples to {samples_file}...")
os.makedirs("/content", exist_ok=True)
with open(samples_file, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("[*] Running EvalPlus MBPP Grader...")
os.system(f"evalplus.evaluate --dataset mbpp --samples {samples_file} --i-just-wanna-run")

results_file = samples_file.replace(".jsonl", "_eval_results.json")
pass_at_1 = 0.0
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        res_data = json.load(f)
        pass_at_1 = res_data["pass@1"]["mbpp"] if "pass@1" in res_data and "mbpp" in res_data["pass@1"] else res_data.get("pass@1", 0.0)
else:
    print(f"[!] Warning: EvalPlus results file not found at {results_file}")

s_per_it = elapsed_time / len(mbpp_dataset)
t_per_s = total_generated_tokens / elapsed_time

update_metrics({
    "INT4": {
        "MBPP Pass@1": pass_at_1,
        "MBPP Time (s)": elapsed_time,
        "MBPP s/it": s_per_it,
        "MBPP t/s (est)": t_per_s,
        "MBPP Power (W)": avg_power,
        "MBPP Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 MBPP Results ---")
print(f"Total Samples: {len(mbpp_dataset)}")
print(f"Total Time: {elapsed_time:.2f} s")
print(f"Pass@1:     {pass_at_1:.4f}")
print(f"Latency:    {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int4
clear_vram()

# %%
# Cell 10: INT4 Dynamic (NF4) Perplexity & Logits
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
    dataset = load_dataset("openai_humaneval", split="test")
    valid_texts = [text for text in dataset["prompt"] if len(text.strip()) > 0][:10]
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    torch.save(outputs.logits.cpu(), filename)
    print(f"[*] Saved dataset logits to {filename} (Shape: {outputs.logits.shape})")

def run_ppl_eval(model, tokenizer, batch_size="auto"):
    print(f"[*] Initializing lm-eval HFLM wrapper for {TASK}...")
    TASK = "wikitext"
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    print(f"[*] Running {TASK} evaluation...")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[TASK],
        num_fewshot=0,
        log_samples=False
    )
    ppl = results["results"][TASK].get("word_perplexity,none", results["results"][TASK].get("word_perplexity", 0.0))
    return ppl

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
print("CELL 10: INT4 Dynamic Perplexity & Logits")
print("="*60)

clear_vram()
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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

print("[*] Starting Hardware Telemetry...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

ppl_score = run_ppl_eval(model_int4, tokenizer)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

save_dataset_logits(model_int4, tokenizer, "qwen_int4_logits.pt")

update_metrics({
    "INT4": {
        "Perplexity": ppl_score,
        "PPL Time (s)": elapsed_time,
        "PPL Power (W)": avg_power,
        "PPL Peak VRAM (GB)": peak_vram
    }
})

print("\n--- INT4 Perplexity Results ---")
print(f"Perplexity: {ppl_score:.4f}")
print(f"Avg Power:  {avg_power:.2f} W")
print(f"Peak VRAM:  {peak_vram:.2f} GB")

del model_int4
clear_vram()

# %%
# Cell 11: KL Divergence & Final Health Card
import json
import torch
import torch.nn.functional as F

def compute_kl(ref_logits, quant_logits):
    if ref_logits.shape != quant_logits.shape:
        min_len = min(ref_logits.shape[1], quant_logits.shape[1])
        ref_logits = ref_logits[:, :min_len, :]
        quant_logits = quant_logits[:, :min_len, :]
    
    p = F.softmax(ref_logits, dim=-1)
    log_q = F.log_softmax(quant_logits, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

print("[*] Loading Logit Tensors...")
fp16_logits = torch.load("qwen_fp16_logits.pt")
int8_logits = torch.load("qwen_int8_logits.pt")
int4_logits = torch.load("qwen_int4_logits.pt")

print("[*] Computing KL Divergence shifts...")
kl_fp16_int8 = compute_kl(fp16_logits, int8_logits)
kl_fp16_int4 = compute_kl(fp16_logits, int4_logits)

with open("qwen_pipeline_metrics.json", "r") as f:
    final_metrics = json.load(f)

final_metrics["FP16"]["KL Divergence"] = 0.0
final_metrics["INT8"]["KL Divergence"] = kl_fp16_int8
final_metrics["INT4"]["KL Divergence"] = kl_fp16_int4

print("\n" + "="*145)
print(f"{'FINAL HEALTH CARD: QWEN 2.5 CODER 1.5B QUANTIZATION PROFILING':^145}")
print("="*145)
print(f"{'Metric':<25} | {'FP16 (Baseline)':<35} | {'INT8 Dynamic':<35} | {'INT4 (NF4) Dynamic':<35}")
print("-" * 145)

metrics_keys = [
    "HumanEval Pass@1", "HumanEval s/it", "HumanEval t/s (est)", "HumanEval Peak VRAM (GB)", 
    "MBPP Pass@1", "MBPP s/it", "MBPP t/s (est)", "MBPP Peak VRAM (GB)", 
    "Perplexity", "PPL Peak VRAM (GB)", "KL Divergence"
]

for key in metrics_keys:
    v1 = final_metrics.get("FP16", {}).get(key, 'N/A')
    v2 = final_metrics.get("INT8", {}).get(key, 'N/A')
    v3 = final_metrics.get("INT4", {}).get(key, 'N/A')
    
    v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
    v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
    v3_str = f"{v3:.4f}" if isinstance(v3, float) else str(v3)
    
    print(f"{key:<25} | {v1_str:<35} | {v2_str:<35} | {v3_str:<35}")

print("="*145)
print("[*] Qwen Pipeline execution finished successfully.")


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

print("\n" + "="*60)
print("CELL 13: Experimental INT8 HumanEval Batched Generation")
print("="*60)

clear_vram()

print("[*] Loading Tokenizer...")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
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
        
        stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
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
        f.write(json.dumps(sample) + "\n")

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

print("\n--- INT8 HumanEval Batched Results ---")
print(f"Total Samples: {len(dataset)}")
print(f"Batch Size:    {BATCH_SIZE}")
print(f"Total Time:    {elapsed_time:.2f} s")
print(f"Pass@1:        {pass_at_1:.4f}")
print(f"Latency:       {s_per_it:.2f} s/it | {t_per_s:.2f} t/s")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model_int8
clear_vram()
