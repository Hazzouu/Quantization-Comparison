import os

gsm8k_mmlu_kl = """
# %% [CELL 4: KL DIVERGENCE FP16 LOGITS]
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "{MODEL_ID}"
print("\\n" + "="*60)
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

torch.save(outputs.logits.cpu(), "{SCRIPT_NAME}_fp16_logits.pt")
print("[*] Saved FP16 logits.")

del model_fp16
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 5: KL DIVERGENCE QLORA & CALCULATION]
from transformers import BitsAndBytesConfig
from peft import PeftModel

print("\\n" + "="*60)
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
model_qlora = PeftModel.from_pretrained(base_model, "./output/{SCRIPT_NAME}")

print("[*] Extracting KL logits for QLoRA model...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
valid_texts = [text for text in dataset["text"] if len(text.strip()) > 0][:100]
inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    outputs_qlora = model_qlora(**inputs)

torch.save(outputs_qlora.logits.cpu(), "{SCRIPT_NAME}_qlora_logits.pt")
print("[*] Saved QLoRA logits.")

del model_qlora, base_model
gc.collect()
torch.cuda.empty_cache()

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("{SCRIPT_NAME}_fp16_logits.pt")
qlora_logits = torch.load("{SCRIPT_NAME}_qlora_logits.pt")

def compute_kl(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits)
print(f"\\n[EVAL] KL Divergence: {kl_div:.4f}\\n")
"""

mbpp_he_kl = """
# %% [CELL 4: KL DIVERGENCE FP16 LOGITS]
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "{MODEL_ID}"
print("\\n" + "="*60)
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

print("[*] Extracting HumanEval Logits (100 samples) for KL symmetry...")
he_dataset = load_dataset("openai_humaneval", split="test").select(range(100))
kl_prompts = [str(p) for p in he_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    kl_outputs = model_fp16(**kl_inputs)

torch.save(kl_outputs.logits.cpu(), "{SCRIPT_NAME}_fp16_logits.pt")
print("[*] Saved FP16 logits.")

del model_fp16
gc.collect()
torch.cuda.empty_cache()

# %% [CELL 5: KL DIVERGENCE QLORA & CALCULATION]
from transformers import BitsAndBytesConfig
from peft import PeftModel

print("\\n" + "="*60)
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
model_qlora = PeftModel.from_pretrained(base_model, "./output/{SCRIPT_NAME}")

print("[*] Extracting KL logits for QLoRA model...")
he_dataset = load_dataset("openai_humaneval", split="test").select(range(100))
kl_prompts = [str(p) for p in he_dataset["prompt"]]
kl_inputs = tokenizer(kl_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")

with torch.no_grad():
    kl_outputs_qlora = model_qlora(**kl_inputs)

torch.save(kl_outputs_qlora.logits.cpu(), "{SCRIPT_NAME}_qlora_logits.pt")
print("[*] Saved QLoRA logits.")

del model_qlora, base_model
gc.collect()
torch.cuda.empty_cache()

print("[*] Loading Logits from disk...")
fp16_logits = torch.load("{SCRIPT_NAME}_fp16_logits.pt")
qlora_logits = torch.load("{SCRIPT_NAME}_qlora_logits.pt")

def compute_kl(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits)
print(f"\\n[EVAL] KL Divergence: {kl_div:.4f}\\n")
"""

scripts = [
    {"name": "sft_llama_3_2_gsm8k.py", "model": "meta-llama/Llama-3.2-1B-Instruct", "template": gsm8k_mmlu_kl},
    {"name": "sft_llama_3_2_mmlu.py", "model": "meta-llama/Llama-3.2-1B-Instruct", "template": gsm8k_mmlu_kl},
    {"name": "sft_llama_3_8b_gsm8k.py", "model": "meta-llama/Meta-Llama-3-8B-Instruct", "template": gsm8k_mmlu_kl},
    {"name": "sft_llama_3_8b_mmlu.py", "model": "meta-llama/Meta-Llama-3-8B-Instruct", "template": gsm8k_mmlu_kl},
    {"name": "sft_qwen_1_5b_mbpp.py", "model": "Qwen/Qwen2.5-Coder-1.5B", "template": mbpp_he_kl},
    {"name": "sft_qwen_7b_mbpp.py", "model": "Qwen/Qwen2.5-Coder-7B", "template": mbpp_he_kl},
]

for script in scripts:
    path = os.path.join(r"c:\Users\youse\Documents\Thesis3", script["name"])
    if os.path.exists(path):
        # We assume that CELL 4 and 5 haven't been appended yet.
        # We just append to the file.
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if "# %% [CELL 4: KL DIVERGENCE FP16 LOGITS]" in content:
            print(f"CELL 4 already exists in {script['name']}, skipping.")
            continue
            
        new_cells = script["template"].replace("{MODEL_ID}", script["model"]).replace("{SCRIPT_NAME}", script["name"].replace(".py", ""))
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(new_cells)
        print(f"Appended KL divergence cells to {script['name']}")
    else:
        print(f"File {path} not found")
