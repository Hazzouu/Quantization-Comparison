import re
import os

gsm8k_eval = """# %% [CELL 3: PHASE 2 EVALUATION]
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
from peft import PeftModel

MODEL_ID = "{MODEL_ID}"

pynvml.nvmlInit()
try:
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
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
    return tps, latency

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    print("[*] Loading wikitext-2-raw-v1 test split for Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\\n\\n".join(dataset["text"]), return_tensors="pt")
    
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

print("\\n" + "="*60)
print("CELL 3: PHASE 2 EVALUATION (SFT LORA)")
print("="*60)

print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading Base Model in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=quant_config, 
    device_map="cuda"
)

print("[*] Attaching LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, "./output/{SCRIPT_NAME}")

print("[*] Starting Hardware Telemetry & Task Evaluation...")
tracker = PowerTracker()
tracker.start()

try:
    acc = run_gsm8k_eval(model, tokenizer)
except Exception as e:
    print("Error in GSM8K Eval:", e)
    acc = 0.0

try:
    ppl = calculate_perplexity(model, tokenizer)
except Exception as e:
    print("Error in PPL Eval:", e)
    ppl = 0.0
    
try:
    tps, s_per_it = measure_generative_latency(model, tokenizer)
except Exception as e:
    print("Error in Generative Latency Eval:", e)
    tps, s_per_it = 0.0, 0.0

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\\n--- LoRA SFT Results ---")
print(f"GSM8K Exact Match: {acc:.4f}")
print(f"Perplexity:        {ppl:.4f}")
print(f"Gen Latency:       {tps:.2f} t/s")
print(f"Latency s/it:      {s_per_it:.4f} s/it")
print(f"Avg Power:         {avg_power:.2f} W")
print(f"Peak VRAM:         {peak_vram:.2f} GB")

del model, base_model
gc.collect()
torch.cuda.empty_cache()
"""

mmlu_eval = """# %% [CELL 3: PHASE 2 EVALUATION]
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
from peft import PeftModel

MODEL_ID = "{MODEL_ID}"

pynvml.nvmlInit()
try:
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
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
    return tps, latency

def calculate_perplexity(model, tokenizer, device="cuda", stride=512):
    print("[*] Loading wikitext-2-raw-v1 test split for Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\\n\\n".join(dataset["text"]), return_tensors="pt")
    
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

def run_mmlu_eval(model, tokenizer, batch_size="auto"):
    print("[*] Initializing lm-eval HFLM wrapper for MMLU...")
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

print("\\n" + "="*60)
print("CELL 3: PHASE 2 EVALUATION (SFT LORA)")
print("="*60)

print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading Base Model in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=quant_config, 
    device_map="cuda"
)

print("[*] Attaching LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, "./output/{SCRIPT_NAME}")

print("[*] Starting Hardware Telemetry & Task Evaluation...")
tracker = PowerTracker()
tracker.start()

try:
    acc = run_mmlu_eval(model, tokenizer)
except Exception as e:
    print("Error in MMLU Eval:", e)
    acc = 0.0

try:
    ppl = calculate_perplexity(model, tokenizer)
except Exception as e:
    print("Error in PPL Eval:", e)
    ppl = 0.0

try:
    tps, s_per_it = measure_generative_latency(model, tokenizer)
except Exception as e:
    print("Error in Generative Latency Eval:", e)
    tps, s_per_it = 0.0, 0.0

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\\n--- LoRA SFT Results ---")
print(f"MMLU Accuracy: {acc:.4f}")
print(f"Perplexity:    {ppl:.4f}")
print(f"Gen Latency:   {tps:.2f} t/s")
print(f"Latency s/it:  {s_per_it:.4f} s/it")
print(f"Avg Power:     {avg_power:.2f} W")
print(f"Peak VRAM:     {peak_vram:.2f} GB")

del model, base_model
gc.collect()
torch.cuda.empty_cache()
"""


mbpp_he_eval = """# %% [CELL 3: PHASE 2 EVALUATION]
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
from peft import PeftModel
from evalplus.data import get_mbpp_plus
from human_eval.evaluation import evaluate_functional_correctness

os.environ["HUMAN_EVAL_ALLOW_EXECUTION"] = "1"
os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O ./HumanEval.jsonl.gz')

MODEL_ID = "{MODEL_ID}"

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

print("\\n" + "="*60)
print("CELL 3: PHASE 2 EVALUATION (SFT LORA)")
print("="*60)

print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading Base Model in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map="cuda")

print("[*] Attaching LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, "./output/{SCRIPT_NAME}")

tracker = PowerTracker()
tracker.start()
start_time = time.time()
BATCH_SIZE = 32

# --- MBPP Eval ---
print("[*] Loading MBPP Dataset...")
mbpp_data = list(get_mbpp_plus().items())
mbpp_samples = []
total_generated_tokens = 0

try:
    for i in tqdm(range(0, len(mbpp_data), BATCH_SIZE), desc="MBPP Generation"):
        batch = mbpp_data[i:i + BATCH_SIZE]
        task_ids = [item[0] for item in batch]
        prompts = [item[1]["prompt"] for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        for j, output in enumerate(generated_ids):
            new_tokens = output[input_length:]
            total_generated_tokens += len(new_tokens)
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            stop_words = ["\\nif __name__ == ", "\\nprint("]
            for stop_word in stop_words:
                if stop_word in completion:
                    completion = completion.split(stop_word)[0]
            try:
                # Local isolation exec requirement
                exec("", {})
            except Exception:
                pass
            mbpp_samples.append({"task_id": task_ids[j], "solution": completion})

    samples_file = "mbpp_{SCRIPT_NAME}_samples.jsonl"
    with open(samples_file, "w") as f:
        for sample in mbpp_samples:
            f.write(json.dumps(sample) + "\\n")

    os.system(f"evalplus.sanitize --samples {samples_file}")
    sanitized_file = samples_file.replace(".jsonl", "-sanitized.jsonl")
    os.system(f"evalplus.evaluate --dataset mbpp --samples {sanitized_file} --i-just-wanna-run")

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
        if total_tasks > 0:
            base_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("base_status") == "pass")
            plus_passes = sum(1 for runs in eval_data.values() if runs and runs[0].get("plus_status") == "pass")
            base_pass_1 = base_passes / total_tasks
            plus_pass_1 = plus_passes / total_tasks
except Exception as e:
    print("Error in MBPP Eval:", e)
    base_pass_1 = 0.0
    plus_pass_1 = 0.0

# --- HumanEval Eval ---
print("[*] Loading HumanEval Dataset...")
he_dataset = load_dataset("openai_humaneval", split="test")
he_samples = []

try:
    for i in tqdm(range(0, len(he_dataset), BATCH_SIZE), desc="HumanEval Generation"):
        batch = he_dataset[i:i + BATCH_SIZE]
        prompts = batch["prompt"]
        task_ids = batch["task_id"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        for j, output in enumerate(generated_ids):
            new_tokens = output[input_length:]
            total_generated_tokens += len(new_tokens)
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            stop_words = ["\\ndef ", "\\nclass ", "\\nif __name__ == ", "\\nprint("]
            for stop_word in stop_words:
                if stop_word in completion:
                    completion = completion.split(stop_word)[0]
            try:
                # Local isolation exec requirement
                exec("", {})
            except Exception:
                pass
            he_samples.append({"task_id": task_ids[j], "completion": completion})

    he_samples_file = "humaneval_{SCRIPT_NAME}_samples.jsonl"
    with open(he_samples_file, "w") as f:
        for sample in he_samples:
            f.write(json.dumps(sample) + "\\n")

    results = evaluate_functional_correctness(sample_file=he_samples_file, k=[1], problem_file="./HumanEval.jsonl.gz")
    he_pass_at_1 = results.get("pass@1", 0.0)
except Exception as e:
    print("Error in HumanEval Eval:", e)
    he_pass_at_1 = 0.0

elapsed_time = time.time() - start_time
total_items = len(mbpp_data) + len(he_dataset)
t_per_s = total_generated_tokens / elapsed_time if elapsed_time > 0 else 0.0
s_per_it = elapsed_time / total_items if total_items > 0 else 0.0

# --- PPL ---
print("[*] Running WikiText-2 PPL Evaluation...")
try:
    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wiki_encodings = tokenizer("\\n\\n".join(wiki_data["text"]), return_tensors="pt")
    max_length = min(model.config.max_position_embeddings, 4096)
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
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    ppl_val = torch.exp(torch.stack(nlls).mean()).item()
except Exception as e:
    print("Error in PPL Eval:", e)
    ppl_val = 0.0

avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("\\n--- LoRA SFT Results ---")
print(f"HumanEval Pass@1: {he_pass_at_1:.4f}")
print(f"MBPP Base Pass@1: {base_pass_1:.4f}")
print(f"MBPP+ Pass@1:     {plus_pass_1:.4f}")
print(f"PPL:              {ppl_val:.4f}")
print(f"Time:             {elapsed_time:.2f} s")
print(f"Tokens/Sec:       {t_per_s:.2f} t/s")
print(f"Latency s/it:     {s_per_it:.4f} s/it")
print(f"Avg Power:        {avg_power:.2f} W")
print(f"Peak VRAM:        {peak_vram:.2f} GB")

del model, base_model
gc.collect()
torch.cuda.empty_cache()
"""

scripts = [
    {"name": "sft_llama_3_2_gsm8k.py", "model": "meta-llama/Llama-3.2-1B-Instruct", "template": gsm8k_eval},
    {"name": "sft_llama_3_2_mmlu.py", "model": "meta-llama/Llama-3.2-1B-Instruct", "template": mmlu_eval},
    {"name": "sft_llama_3_8b_gsm8k.py", "model": "meta-llama/Meta-Llama-3-8B-Instruct", "template": gsm8k_eval},
    {"name": "sft_llama_3_8b_mmlu.py", "model": "meta-llama/Meta-Llama-3-8B-Instruct", "template": mmlu_eval},
    {"name": "sft_qwen_1_5b_mbpp.py", "model": "Qwen/Qwen2.5-Coder-1.5B", "template": mbpp_he_eval},
    {"name": "sft_qwen_7b_mbpp.py", "model": "Qwen/Qwen2.5-Coder-7B", "template": mbpp_he_eval},
]

for script in scripts:
    path = os.path.join(r"c:\Users\youse\Documents\Thesis3", script["name"])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        parts = content.split("# %% [CELL 3: PHASE 2 EVALUATION]")
        if len(parts) > 1:
            new_content = parts[0] + script["template"].replace("{MODEL_ID}", script["model"]).replace("{SCRIPT_NAME}", script["name"].replace(".py", ""))
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {script['name']}")
        else:
            print(f"Cell 3 marker not found in {script['name']}")
    else:
        print(f"File {path} not found")
