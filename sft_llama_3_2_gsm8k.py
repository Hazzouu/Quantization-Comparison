# !pip install trl peft bitsandbytes lm-eval evalplus

# %% [CELL 1: SFT TRAINING]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_id = "meta-llama/Llama-3.2-1B-Instruct"

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

dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(lambda x: {"text": f"Question: {x['question']}\n\nAnswer: {x['answer']}"})

sft_config = SFTConfig(
    output_dir="./output/sft_llama_3_2_gsm8k",
    per_device_train_batch_size=4,
    max_steps=100,
    logging_steps=10,
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

trainer.model.save_pretrained("./output/sft_llama_3_2_gsm8k")

metrics = {
    "Total_Time_s": total_time,
    "Peak_VRAM_GB": peak_vram,
    "Avg_Power_W": avg_power,
    "Train_Loss": train_loss
}
with open("sft_llama_3_2_gsm8k_metrics.json", "w") as f:
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

# %% [CELL 3: PHASE 2 EVALUATION]
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

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

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

print("\n" + "="*60)
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
model = PeftModel.from_pretrained(base_model, "./output/sft_llama_3_2_gsm8k")

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

print("\n--- LoRA SFT Results ---")
print(f"GSM8K Exact Match: {acc:.4f}")
print(f"Perplexity:        {ppl:.4f}")
print(f"Gen Latency:       {tps:.2f} t/s")
print(f"Latency s/it:      {s_per_it:.4f} s/it")
print(f"Avg Power:         {avg_power:.2f} W")
print(f"Peak VRAM:         {peak_vram:.2f} GB")

del model, base_model
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 4: KL DIVERGENCE FP16 LOGITS]
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
print("\n" + "="*60)
print("CELL 4: KL DIVERGENCE FP16 LOGITS")
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

print("[*] Extracting KL logits on 100 sequences of WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    outputs = model_fp16(**inputs)

torch.save(outputs.logits.cpu(), "sft_llama_3_2_gsm8k_fp16_logits.pt")
print("[*] Saved FP16 logits.")

del model_fp16
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 5: KL DIVERGENCE QLORA & CALCULATION]
from transformers import BitsAndBytesConfig
from peft import PeftModel

print("\n" + "="*60)
print("CELL 5: KL DIVERGENCE QLORA & CALCULATION")
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
model_qlora = PeftModel.from_pretrained(base_model, "./output/sft_llama_3_2_gsm8k")

print("[*] Extracting KL logits for QLoRA model...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    outputs_qlora = model_qlora(**inputs)

torch.save(outputs_qlora.logits.cpu(), "sft_llama_3_2_gsm8k_qlora_logits.pt")
print("[*] Saved QLoRA logits.")

del model_qlora, base_model
gc.collect()
torch.cuda.empty_cache()

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("sft_llama_3_2_gsm8k_fp16_logits.pt")
qlora_logits = torch.load("sft_llama_3_2_gsm8k_qlora_logits.pt")

def compute_kl(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits)
print(f"\n[EVAL] KL Divergence: {kl_div:.4f}\n")
