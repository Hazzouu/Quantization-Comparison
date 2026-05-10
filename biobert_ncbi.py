# %% [markdown]
# # BioBERT PTQ Pipeline (NCBI Disease)
# Complete Post-Training Quantization Evaluation Pipeline for dmis-lab/biobert-v1.1

# %%
# Cell 1: Package Installations
# !pip install transformers datasets evaluate scikit-learn accelerate bitsandbytes pynvml torch seqeval safetensors

# %%
# Cell 2: FP16 Baseline Evaluation & Logit/Activation Extraction
import gc
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pynvml
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from safetensors.torch import save_file
from datasets import load_dataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    gc.collect()
    time.sleep(2) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass 
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 2: FP16 Baseline Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "alvaroalon2/biobert_genetic_ner"
DATASET_NAME = "ncbi_disease"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset_dict = load_dataset(DATASET_NAME)
val_split = dataset_dict["validation"]

train_split = dataset_dict["train"]
tag_col = "ner_tags" if "ner_tags" in train_split.features else "tags"
label_list = train_split.features[tag_col].feature.names
num_labels = len(label_list)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples[tag_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = val_split.map(tokenize_and_align_labels, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print("[*] Loading FP16 Model...")
model_fp16 = AutoModelForTokenClassification.from_pretrained(MODEL_ID, num_labels=num_labels, torch_dtype=torch.float16, device_map="cuda")

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
    
    h5 = model.bert.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.bert.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    
    print(f"[*] Model Dictionary: {model.config.id2label}")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            
            predictions = preds.cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            # Ensure model's id2label has integer keys to prevent KeyError
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            # The NCBI Disease Dictionary
            dataset_id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

            for i in range(labels.shape[0]):
                # 1. Ground truth perfectly mapped via dataset dictionary
                true_labels = [dataset_id2label[int(l)] for l in labels[i] if l != -100]
                
                # 2. Extract model's native strings via its own config
                raw_preds = [id2label[int(p)] for (p, l) in zip(predictions[i], labels[i]) if l != -100]
                
                # 3. The String Normalizer: Force model strings to match NCBI format
                preds = []
                for rp in raw_preds:
                    rp_upper = rp.upper()
                    if rp_upper.startswith("B"): 
                        preds.append("B-Disease")
                    elif rp_upper.startswith("I"): 
                        preds.append("I-Disease")
                    else: 
                        preds.append("O")
                        
                all_labels.append(true_labels)
                all_preds.append(preds)
            
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

print("[*] Computing Seqeval Metrics...")
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- FP16 Results ---")
print(f"Accuracy:      {acc:.4f}")
print(f"Macro F1:      {f1:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
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
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, BitsAndBytesConfig
from safetensors.torch import save_file
from datasets import load_dataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    gc.collect()
    time.sleep(2) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass 
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 3: INT8 Dynamic Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "alvaroalon2/biobert_genetic_ner"
DATASET_NAME = "ncbi_disease"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset_dict = load_dataset(DATASET_NAME)
val_split = dataset_dict["validation"]

train_split = dataset_dict["train"]
tag_col = "ner_tags" if "ner_tags" in train_split.features else "tags"
label_list = train_split.features[tag_col].feature.names
num_labels = len(label_list)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples[tag_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = val_split.map(tokenize_and_align_labels, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print("[*] Loading INT8 Model...")
quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForTokenClassification.from_pretrained(MODEL_ID, num_labels=num_labels, quantization_config=quant_config_8bit, device_map="cuda")

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
    
    h5 = model.bert.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.bert.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    
    print(f"[*] Model Dictionary: {model.config.id2label}")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            
            predictions = preds.cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            # Ensure model's id2label has integer keys to prevent KeyError
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            # The NCBI Disease Dictionary
            dataset_id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

            for i in range(labels.shape[0]):
                # 1. Ground truth perfectly mapped via dataset dictionary
                true_labels = [dataset_id2label[int(l)] for l in labels[i] if l != -100]
                
                # 2. Extract model's native strings via its own config
                raw_preds = [id2label[int(p)] for (p, l) in zip(predictions[i], labels[i]) if l != -100]
                
                # 3. The String Normalizer: Force model strings to match NCBI format
                preds = []
                for rp in raw_preds:
                    rp_upper = rp.upper()
                    if rp_upper.startswith("B"): 
                        preds.append("B-Disease")
                    elif rp_upper.startswith("I"): 
                        preds.append("I-Disease")
                    else: 
                        preds.append("O")
                        
                all_labels.append(true_labels)
                all_preds.append(preds)
            
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

print("[*] Computing Seqeval Metrics...")
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- INT8 Results ---")
print(f"Accuracy:      {acc:.4f}")
print(f"Macro F1:      {f1:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
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
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, BitsAndBytesConfig
from safetensors.torch import save_file
from datasets import load_dataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    gc.collect()
    time.sleep(2) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass 
    print("[*] VRAM cleared successfully.")

print("\n" + "="*60)
print("CELL 4: INT4 Dynamic Evaluation")
print("="*60)
clear_vram()

MODEL_ID = "alvaroalon2/biobert_genetic_ner"
DATASET_NAME = "ncbi_disease"
print("[*] Loading Tokenizer & Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset_dict = load_dataset(DATASET_NAME)
val_split = dataset_dict["validation"]

train_split = dataset_dict["train"]
tag_col = "ner_tags" if "ner_tags" in train_split.features else "tags"
label_list = train_split.features[tag_col].feature.names
num_labels = len(label_list)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples[tag_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = val_split.map(tokenize_and_align_labels, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print("[*] Loading INT4 Model...")
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_int4 = AutoModelForTokenClassification.from_pretrained(MODEL_ID, num_labels=num_labels, quantization_config=quant_config_4bit, device_map="cuda")

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
    
    h5 = model.bert.encoder.layer[5].output.dense.register_forward_hook(get_hook(layer5_acts))
    h11 = model.bert.encoder.layer[11].output.dense.register_forward_hook(get_hook(layer11_acts))
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    
    print(f"[*] Model Dictionary: {model.config.id2label}")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_logits.append(logits.detach().cpu())
            
            predictions = preds.cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            # Ensure model's id2label has integer keys to prevent KeyError
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            # The NCBI Disease Dictionary
            dataset_id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

            for i in range(labels.shape[0]):
                # 1. Ground truth perfectly mapped via dataset dictionary
                true_labels = [dataset_id2label[int(l)] for l in labels[i] if l != -100]
                
                # 2. Extract model's native strings via its own config
                raw_preds = [id2label[int(p)] for (p, l) in zip(predictions[i], labels[i]) if l != -100]
                
                # 3. The String Normalizer: Force model strings to match NCBI format
                preds = []
                for rp in raw_preds:
                    rp_upper = rp.upper()
                    if rp_upper.startswith("B"): 
                        preds.append("B-Disease")
                    elif rp_upper.startswith("I"): 
                        preds.append("I-Disease")
                    else: 
                        preds.append("O")
                        
                all_labels.append(true_labels)
                all_preds.append(preds)
            
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

print("[*] Computing Seqeval Metrics...")
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")
latency = (elapsed_time / len(all_labels)) * 1000

print("\n--- INT4 Results ---")
print(f"Accuracy:      {acc:.4f}")
print(f"Macro F1:      {f1:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
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
from safetensors.torch import load_file

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
    
    # Flatten [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
    # This guarantees we get the true Average KL per token
    flat_fp16 = fp16_logits.view(-1, fp16_logits.size(-1))
    flat_int8 = int8_logits.view(-1, int8_logits.size(-1))
    flat_int4 = int4_logits.view(-1, int4_logits.size(-1))
    
    # Calculate KL Divergence
    int8_q_log_probs = F.log_softmax(flat_int8, dim=-1)
    int4_q_log_probs = F.log_softmax(flat_int4, dim=-1)
    p_probs = F.softmax(flat_fp16, dim=-1)
    
    kl_int8 = F.kl_div(int8_q_log_probs, p_probs, reduction='batchmean').item()
    kl_int4 = F.kl_div(int4_q_log_probs, p_probs, reduction='batchmean').item()
    
    print("\n--- KL Divergence vs FP16 ---")
    print(f"INT8 Drift: {kl_int8:.6f}")
    print(f"INT4 Drift: {kl_int4:.6f}")
else:
    print("[!] Logits not found. Run Cells 2-4 first.")
