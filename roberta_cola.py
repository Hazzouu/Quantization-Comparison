# %% [markdown]
# # RoBERTa-base PTQ Pipeline (CoLA)
# Complete Post-Training Quantization Evaluation Pipeline for textattack/roberta-base-CoLA

# %%
# Cell 1: Package Installations
# !pip install -q -U transformers datasets evaluate scikit-learn accelerate bitsandbytes pynvml torch

# %%
# Cell 2: FP16 Baseline Evaluation & Logit/Activation Extraction
import gc
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from safetensors.torch import save_file
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from accelerate import find_executable_batch_size
from torch.utils.data import DataLoader

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
    # 1. Clear standard Python garbage
    gc.collect()
    time.sleep(2) 
    
    # 2. Check if a GPU is available in the environment
    if torch.cuda.is_available():
        # 3. Actually release the cached memory back to the system
        torch.cuda.empty_cache()
        
        # 4. Only reset peak stats if the CUDA context has been initialized
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass # Failsafe just in case the allocator is still asleep
                
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 2: FP16 Baseline Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "textattack/roberta-base-CoLA"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("glue", "cola", split="validation")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")

layer5_acts = []
layer11_acts = []

def get_hook(layer_list):
    def hook(module, input, output):
        if len(layer_list) == 0:
            layer_list.append(output.detach().cpu())
    return hook

@find_executable_batch_size(starting_batch_size=256)
def eval_loop(batch_size, model, layer5_acts, layer11_acts):
    print(f"[*] Trying Batch Size: {batch_size}")
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    layer5_acts.clear()
    layer11_acts.clear()
    
    h5 = model.roberta.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.roberta.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels" if "labels" in batch else "label"].cpu().numpy())
            
    h5.remove()
    h11.remove()
    
    return all_logits, all_preds, all_labels

print("[*] Starting Hardware Telemetry & Batched Inference...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

fp16_all_logits, all_preds, all_labels = eval_loop(model_fp16, layer5_acts, layer11_acts)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

mcc_score = matthews_corrcoef(all_labels, all_preds)
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- FP16 Results ---")
print(f"MCC:           {mcc_score:.4f}")
print(f"Latency:       {latency:.2f} ms/sample")
print(f"Peak VRAM:     {peak_vram:.2f} GB")
print(f"Avg Power:     {avg_power:.2f} W")

print("[*] Saving FP16 Logits & Activations...")
fp16_logits_tensor = torch.cat(fp16_all_logits, dim=0)
save_file({"logits": fp16_logits_tensor}, "fp16_logits.safetensors")
save_file({"acts": layer5_acts[0]}, "fp16_acts_layer5.safetensors")
save_file({"acts": layer11_acts[0]}, "fp16_acts_layer11.safetensors")

del model_fp16
clear_vram()

# %%
# Cell 3: INT8 Dynamic Evaluation & Logit/Activation Extraction
import gc
import time
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig
from safetensors.torch import save_file
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from accelerate import find_executable_batch_size
from torch.utils.data import DataLoader

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
    # 1. Clear standard Python garbage
    gc.collect()
    time.sleep(2) 
    
    # 2. Check if a GPU is available in the environment
    if torch.cuda.is_available():
        # 3. Actually release the cached memory back to the system
        torch.cuda.empty_cache()
        
        # 4. Only reset peak stats if the CUDA context has been initialized
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass # Failsafe just in case the allocator is still asleep
                
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 3: INT8 Dynamic Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "textattack/roberta-base-CoLA"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("glue", "cola", split="validation")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, quantization_config=quant_config_8bit, device_map="cuda")

layer5_acts = []
layer11_acts = []

def get_hook(layer_list):
    def hook(module, input, output):
        if len(layer_list) == 0:
            layer_list.append(output.detach().cpu())
    return hook

@find_executable_batch_size(starting_batch_size=256)
def eval_loop_int8(batch_size, model, layer5_acts, layer11_acts):
    print(f"[*] Trying Batch Size: {batch_size}")
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    layer5_acts.clear()
    layer11_acts.clear()
    
    h5 = model.roberta.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.roberta.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels" if "labels" in batch else "label"].cpu().numpy())
            
    h5.remove()
    h11.remove()
    
    return all_logits, all_preds, all_labels

print("[*] Starting Hardware Telemetry & Batched Inference...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

int8_all_logits, all_preds, all_labels = eval_loop_int8(model_int8, layer5_acts, layer11_acts)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

mcc_score = matthews_corrcoef(all_labels, all_preds)
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- INT8 Results ---")
print(f"MCC:           {mcc_score:.4f}")
print(f"Latency:       {latency:.2f} ms/sample")
print(f"Peak VRAM:     {peak_vram:.2f} GB")
print(f"Avg Power:     {avg_power:.2f} W")

print("[*] Saving INT8 Logits & Activations...")
int8_logits_tensor = torch.cat(int8_all_logits, dim=0)
save_file({"logits": int8_logits_tensor}, "int8_logits.safetensors")
save_file({"acts": layer5_acts[0]}, "int8_acts_layer5.safetensors")
save_file({"acts": layer11_acts[0]}, "int8_acts_layer11.safetensors")

del model_int8
clear_vram()

# %%
# Cell 4: INT4 Dynamic Evaluation & Logit/Activation Extraction
import gc
import time
import threading
import numpy as np
import torch
import pynvml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig
from safetensors.torch import save_file
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from accelerate import find_executable_batch_size
from torch.utils.data import DataLoader

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
    # 1. Clear standard Python garbage
    gc.collect()
    time.sleep(2) 
    
    # 2. Check if a GPU is available in the environment
    if torch.cuda.is_available():
        # 3. Actually release the cached memory back to the system
        torch.cuda.empty_cache()
        
        # 4. Only reset peak stats if the CUDA context has been initialized
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass # Failsafe just in case the allocator is still asleep
                
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 4: INT4 Dynamic Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "textattack/roberta-base-CoLA"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset("glue", "cola", split="validation")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("[*] Loading INT4 Model...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, quantization_config=quant_config_4bit, device_map="cuda")

layer5_acts = []
layer11_acts = []

def get_hook(layer_list):
    def hook(module, input, output):
        if len(layer_list) == 0:
            layer_list.append(output.detach().cpu())
    return hook

@find_executable_batch_size(starting_batch_size=256)
def eval_loop_int4(batch_size, model, layer5_acts, layer11_acts):
    print(f"[*] Trying Batch Size: {batch_size}")
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    layer5_acts.clear()
    layer11_acts.clear()
    
    h5 = model.roberta.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.roberta.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels" if "labels" in batch else "label"].cpu().numpy())
            
    h5.remove()
    h11.remove()
    
    return all_logits, all_preds, all_labels

print("[*] Starting Hardware Telemetry & Batched Inference...")
tracker = PowerTracker()
tracker.start()
start_time = time.time()

int4_all_logits, all_preds, all_labels = eval_loop_int4(model_int4, layer5_acts, layer11_acts)

elapsed_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

mcc_score = matthews_corrcoef(all_labels, all_preds)
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- INT4 Results ---")
print(f"MCC:           {mcc_score:.4f}")
print(f"Latency:       {latency:.2f} ms/sample")
print(f"Peak VRAM:     {peak_vram:.2f} GB")
print(f"Avg Power:     {avg_power:.2f} W")

print("[*] Saving INT4 Logits & Activations...")
int4_logits_tensor = torch.cat(int4_all_logits, dim=0)
save_file({"logits": int4_logits_tensor}, "int4_logits.safetensors")
save_file({"acts": layer5_acts[0]}, "int4_acts_layer5.safetensors")
save_file({"acts": layer11_acts[0]}, "int4_acts_layer11.safetensors")

del model_int4
clear_vram()

# %%
# Cell 5: Deep Degradation: SQNR Analysis
import os
import torch

print("\n" + "="*60)
print("CELL 5: Deep Degradation - SQNR Analysis")
print("="*60)

def compute_sqnr(fp16_act, quant_act):
    signal_power = torch.sum(fp16_act**2)
    noise_power = torch.sum((fp16_act - quant_act)**2)
    if noise_power == 0:
        return float('inf')
    return 10 * torch.log10(signal_power / noise_power).item()

if os.path.exists("fp16_acts_layer5.safetensors"):
    print("[*] Loading Activations...")
    fp16_l5 = load_file("fp16_acts_layer5.safetensors")["acts"].float()
    fp16_l11 = load_file("fp16_acts_layer11.safetensors")["acts"].float()
    
    int8_l5 = load_file("int8_acts_layer5.safetensors")["acts"].float()
    int8_l11 = load_file("int8_acts_layer11.safetensors")["acts"].float()
    
    int4_l5 = load_file("int4_acts_layer5.safetensors")["acts"].float()
    int4_l11 = load_file("int4_acts_layer11.safetensors")["acts"].float()
    
    print("\n--- INT8 vs FP16 SQNR ---")
    print(f"Layer 5 (Mid):     {compute_sqnr(fp16_l5, int8_l5):.2f} dB")
    print(f"Layer 11 (Final):  {compute_sqnr(fp16_l11, int8_l11):.2f} dB")
    
    print("\n--- INT4 vs FP16 SQNR ---")
    print(f"Layer 5 (Mid):     {compute_sqnr(fp16_l5, int4_l5):.2f} dB")
    print(f"Layer 11 (Final):  {compute_sqnr(fp16_l11, int4_l11):.2f} dB")
else:
    print("[!] Activation tensors not found. Run Cells 2-4 first.")

# %%
# Cell 6: Deep Degradation: KL Divergence Analysis
import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

print("\n" + "="*60)
print("CELL 6: Deep Degradation - KL Divergence")
print("="*60)

if os.path.exists("fp16_logits.safetensors"):
    print("[*] Loading Logits...")
    fp16_logits = load_file("fp16_logits.safetensors")["logits"].float()
    int8_logits = load_file("int8_logits.safetensors")["logits"].float()
    int4_logits = load_file("int4_logits.safetensors")["logits"].float()
    
    # Calculate Average KL Divergence
    int8_q_log_probs = F.log_softmax(int8_logits, dim=-1)
    int4_q_log_probs = F.log_softmax(int4_logits, dim=-1)
    p_probs = F.softmax(fp16_logits, dim=-1)
    
    kl_int8 = F.kl_div(int8_q_log_probs, p_probs, reduction='batchmean').item()
    kl_int4 = F.kl_div(int4_q_log_probs, p_probs, reduction='batchmean').item()
    
    print("\n--- KL Divergence vs FP16 ---")
    print(f"INT8 Drift: {kl_int8:.6f}")
    print(f"INT4 Drift: {kl_int4:.6f}")
else:
    print("[!] Logits not found. Run Cells 2-4 first.")
