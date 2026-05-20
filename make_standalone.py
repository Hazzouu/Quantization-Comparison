import os

def make_cells_standalone(file_path, model_name, script_name):
    if not os.path.exists(file_path):
        return

    standalone_header = f"""import os
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

MODEL_ID = "{model_name}"

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
model = PeftModel.from_pretrained(base_model, "./output/{script_name}")
BATCH_SIZE = 32

"""

    vram_cleanup = """
del model, base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Inject VRAM cleanup to the end of Cell 3
    if "print(f\"Peak VRAM:        {peak_vram_mbpp:.2f} GB\\n\")" in content:
        content = content.replace("print(f\"Peak VRAM:        {peak_vram_mbpp:.2f} GB\\n\")", 
                                  "print(f\"Peak VRAM:        {peak_vram_mbpp:.2f} GB\\n\")\n" + vram_cleanup)

    # Inject Standalone Header to Cell 4
    cell_4_marker = "print(\"\\n\" + \"=\"*60)\nprint(\"CELL 4: PHASE 2 EVALUATION - HUMANEVAL\")\nprint(\"=\"*60)"
    if cell_4_marker in content:
        content = content.replace(cell_4_marker, cell_4_marker + "\n\n" + standalone_header)
        
    # Inject VRAM cleanup to the end of Cell 4
    if "print(f\"Peak VRAM:        {peak_vram_he:.2f} GB\\n\")" in content:
        content = content.replace("print(f\"Peak VRAM:        {peak_vram_he:.2f} GB\\n\")", 
                                  "print(f\"Peak VRAM:        {peak_vram_he:.2f} GB\\n\")\n" + vram_cleanup)

    # Inject Standalone Header to Cell 5
    cell_5_marker = "print(\"\\n\" + \"=\"*60)\nprint(\"CELL 5: PHASE 2 EVALUATION - PPL\")\nprint(\"=\"*60)"
    if cell_5_marker in content:
        content = content.replace(cell_5_marker, cell_5_marker + "\n\n" + standalone_header)
        
    # Cell 5 already has a VRAM cleanup at the end, so we leave it alone.

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Made cells standalone in {script_name}")

rewrite_qwen_script = make_cells_standalone
rewrite_qwen_script(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py", "Qwen/Qwen2.5-Coder-7B", "sft_qwen_7b_mbpp")
rewrite_qwen_script(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py", "Qwen/Qwen2.5-Coder-1.5B", "sft_qwen_1_5b_mbpp")
