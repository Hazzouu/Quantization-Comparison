# !pip install trl peft bitsandbytes lm-eval evalplus

# %% [CELL 1: SFT TRAINING]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_id = "Qwen/Qwen2.5-Coder-1.5B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("mbpp", split="train")
dataset = dataset.map(lambda x: {"text": f"Prompt: {x['text']}\n\nCode:\n{x['code']}"})

sft_config = SFTConfig(
    output_dir="./output/sft_qwen_1_5b_mbpp",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=50,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

import time
import json
import threading
import numpy as np
import pynvml

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

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
tracker = PowerTracker()
tracker.start()
start_time = time.time()

trainer.train()

total_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

train_loss = None
for log in trainer.state.log_history:
    if 'loss' in log:
        train_loss = log['loss']

trainer.model.save_pretrained("./output/sft_qwen_1_5b_mbpp")

metrics = {
    "Total_Time_s": total_time,
    "Peak_VRAM_GB": peak_vram,
    "Avg_Power_W": avg_power,
    "Train_Loss": train_loss
}
with open("sft_qwen_1_5b_mbpp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n--- Training Telemetry ---")
print(f"Total Training Time: {total_time:.2f} s")
print(f"Average Power:       {avg_power:.2f} W")
print(f"Peak VRAM:           {peak_vram:.2f} GB")
print(f"Train Loss:          {train_loss}\n")


del trainer, model

# %% [CELL 2: VRAM CLEANUP]
import gc
import torch

print("\n" + "="*60)
print("CELL 2: VRAM CLEANUP")
print("="*60)

# Delete dangling variables from global scope if they exist
for obj in ['model', 'trainer', 'base_model', 'tokenizer', 'inputs', 'outputs']:
    if obj in globals():
        del globals()[obj]

# Run Garbage Collection multiple times to clear cyclic references
gc.collect()
gc.collect()

# Clear CUDA cache and IPC, and reset memory stats
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

print("[*] VRAM cleared and reset successfully.\n")

# %% [CELL 3: PHASE 2 EVALUATION - MBPP]
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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

print("\n" + "="*60)
print("CELL 3: PHASE 2 EVALUATION - MBPP")
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
model = PeftModel.from_pretrained(base_model, "./output/sft_qwen_1_5b_mbpp")

tracker = PowerTracker()
tracker.start()
start_time = time.time()
BATCH_SIZE = 32

print("[*] Loading MBPP Dataset...")
mbpp_data = list(get_mbpp_plus().items())
mbpp_samples = []
total_generated_tokens = 0

try:
    for i in tqdm(range(0, len(mbpp_data), BATCH_SIZE), desc="MBPP Generation"):
        batch = mbpp_data[i:i + BATCH_SIZE]
        task_ids = [item[0] for item in batch]
        prompts = [f"Prompt: Complete the following Python code:\n{item[1]['prompt']}\n\nCode:\n{item[1]['prompt']}" for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        for j, output in enumerate(generated_ids):
            new_tokens = output[input_length:]
            total_generated_tokens += len(new_tokens)
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            stop_words = ["\nif __name__ == ", "\nprint("]
            for stop_word in stop_words:
                if stop_word in completion:
                    completion = completion.split(stop_word)[0]
            try:
                exec("", {})
            except Exception:
                pass
            mbpp_samples.append({"task_id": task_ids[j], "solution": completion})

    samples_file = "mbpp_sft_qwen_1_5b_mbpp_samples.jsonl"
    with open(samples_file, "w") as f:
        for sample in mbpp_samples:
            f.write(json.dumps(sample) + "\n")

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

elapsed_time_mbpp = time.time() - start_time
t_per_s_mbpp = total_generated_tokens / elapsed_time_mbpp if elapsed_time_mbpp > 0 else 0.0
s_per_it_mbpp = elapsed_time_mbpp / len(mbpp_data) if len(mbpp_data) > 0 else 0.0
avg_power_mbpp = tracker.stop()
tracker.join()
peak_vram_mbpp = torch.cuda.max_memory_allocated() / (1024**3)

print("\n--- MBPP SFT Results ---")
print(f"MBPP Base Pass@1: {base_pass_1:.4f}")
print(f"MBPP+ Pass@1:     {plus_pass_1:.4f}")
print(f"Time:             {elapsed_time_mbpp:.2f} s")
print(f"Tokens/Sec:       {t_per_s_mbpp:.2f} t/s")
print(f"Latency s/it:     {s_per_it_mbpp:.4f} s/it")
print(f"Avg Power:        {avg_power_mbpp:.2f} W")
print(f"Peak VRAM:        {peak_vram_mbpp:.2f} GB\n")

del model, base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% [CELL 4: PHASE 2 EVALUATION - HUMANEVAL]
import os
from human_eval.evaluation import evaluate_functional_correctness

print("\n" + "="*60)
print("CELL 4: PHASE 2 EVALUATION - HUMANEVAL")
print("="*60)

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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

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
model = PeftModel.from_pretrained(base_model, "./output/sft_qwen_1_5b_mbpp")
BATCH_SIZE = 32



os.environ["HUMAN_EVAL_ALLOW_EXECUTION"] = "1"
os.system('wget -q -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -O ./HumanEval.jsonl.gz')

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
tracker = PowerTracker()
tracker.start()
start_time = time.time()

print("[*] Loading HumanEval Dataset...")
he_dataset = load_dataset("openai_humaneval", split="test")
he_samples = []
total_generated_tokens = 0

try:
    for i in tqdm(range(0, len(he_dataset), BATCH_SIZE), desc="HumanEval Generation"):
        batch = he_dataset[i:i + BATCH_SIZE]
        prompts = [f"Prompt: Complete the following Python code:\n{p}\n\nCode:\n{p}" for p in batch["prompt"]]
        task_ids = batch["task_id"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        for j, output in enumerate(generated_ids):
            new_tokens = output[input_length:]
            total_generated_tokens += len(new_tokens)
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            stop_words = ["\ndef ", "\nclass ", "\nif __name__ == ", "\nprint("]
            for stop_word in stop_words:
                if stop_word in completion:
                    completion = completion.split(stop_word)[0]
            try:
                exec("", {})
            except Exception:
                pass
            he_samples.append({"task_id": task_ids[j], "completion": completion})

    he_samples_file = "humaneval_sft_qwen_1_5b_mbpp_samples.jsonl"
    with open(he_samples_file, "w") as f:
        for sample in he_samples:
            f.write(json.dumps(sample) + "\n")

    results = evaluate_functional_correctness(sample_file=he_samples_file, k=[1], problem_file="./HumanEval.jsonl.gz")
    he_pass_at_1 = results.get("pass@1", 0.0)
except Exception as e:
    print("Error in HumanEval Eval:", e)
    he_pass_at_1 = 0.0

elapsed_time_he = time.time() - start_time
t_per_s_he = total_generated_tokens / elapsed_time_he if elapsed_time_he > 0 else 0.0
s_per_it_he = elapsed_time_he / len(he_dataset) if len(he_dataset) > 0 else 0.0
avg_power_he = tracker.stop()
tracker.join()
peak_vram_he = torch.cuda.max_memory_allocated() / (1024**3)

print("\n--- HumanEval SFT Results ---")
print(f"HumanEval Pass@1: {he_pass_at_1:.4f}")
print(f"Time:             {elapsed_time_he:.2f} s")
print(f"Tokens/Sec:       {t_per_s_he:.2f} t/s")
print(f"Latency s/it:     {s_per_it_he:.4f} s/it")
print(f"Avg Power:        {avg_power_he:.2f} W")
print(f"Peak VRAM:        {peak_vram_he:.2f} GB\n")

del model, base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% [CELL 5: PHASE 2 EVALUATION - PPL]
print("\n" + "="*60)
print("CELL 5: PHASE 2 EVALUATION - PPL")
print("="*60)

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
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    self.power_readings.append(power_mw / 1000.0)
                except Exception:
                    pass
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        return np.mean(self.power_readings) if self.power_readings else 0.0

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
model = PeftModel.from_pretrained(base_model, "./output/sft_qwen_1_5b_mbpp")
BATCH_SIZE = 32



print("[*] Running WikiText-2 PPL Evaluation...")
try:
    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wiki_encodings = tokenizer("\n\n".join(wiki_data["text"]), return_tensors="pt")
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

print(f"\n[EVAL] PPL: {ppl_val:.4f}\n")

del model, base_model
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 6: KL DIVERGENCE FP16 LOGITS]
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("\n" + "="*60)
print("CELL 6: KL DIVERGENCE FP16 LOGITS")
print("="*60)

print("[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading FP16 Base Model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="cuda"
)

print("[*] Extracting HumanEval Logits (100 samples) for KL symmetry...")
he_dataset = load_dataset("openai_humaneval", split="test").select(range(100))
kl_prompts = [str(p) for p in he_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    kl_outputs = model_fp16(**kl_inputs)

torch.save(kl_outputs.logits.cpu(), "sft_qwen_1_5b_mbpp_fp16_logits.pt")
print("[*] Saved FP16 logits.")

del model_fp16
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 7: KL DIVERGENCE QLORA & CALCULATION]
from transformers import BitsAndBytesConfig
from peft import PeftModel

print("\n" + "="*60)
print("CELL 7: KL DIVERGENCE QLORA & CALCULATION")
print("="*60)

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
model_qlora = PeftModel.from_pretrained(base_model, "./output/sft_qwen_1_5b_mbpp")

print("[*] Extracting KL logits for QLoRA model...")
he_dataset = load_dataset("openai_humaneval", split="test").select(range(100))
kl_prompts = [str(p) for p in he_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    kl_outputs_qlora = model_qlora(**kl_inputs)

torch.save(kl_outputs_qlora.logits.cpu(), "sft_qwen_1_5b_mbpp_qlora_logits.pt")
print("[*] Saved QLoRA logits.")

del model_qlora, base_model
gc.collect()
torch.cuda.empty_cache()

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("sft_qwen_1_5b_mbpp_fp16_logits.pt")
qlora_logits = torch.load("sft_qwen_1_5b_mbpp_qlora_logits.pt")

vocab_size = fp16_logits.size(-1)

def compute_kl(logits_p, logits_q, vocab_size):
    flat_p = logits_p.view(-1, vocab_size)
    flat_q = logits_q.view(-1, vocab_size)
    p_probs = F.softmax(flat_p, dim=-1)
    q_log_probs = F.log_softmax(flat_q, dim=-1)
    kl = F.kl_div(q_log_probs, p_probs, reduction='batchmean').item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits, vocab_size)
print(f"\n[EVAL] KL Divergence: {kl_div:.4f}\n")
