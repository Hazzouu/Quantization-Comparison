import os
import re

new_cell_2 = """# %% [CELL 2: VRAM CLEANUP]
import gc
import torch

print("\\n" + "="*60)
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

print("[*] VRAM cleared and reset successfully.\\n")

# %% [CELL 3: PHASE 2 EVALUATION]"""

scripts = [
    "sft_llama_3_2_gsm8k.py",
    "sft_llama_3_2_mmlu.py",
    "sft_llama_3_8b_gsm8k.py",
    "sft_llama_3_8b_mmlu.py",
    "sft_qwen_1_5b_mbpp.py",
    "sft_qwen_7b_mbpp.py"
]

for script in scripts:
    path = os.path.join(r"c:\Users\youse\Documents\Thesis3", script)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex to replace everything from [CELL 2] up to but not including [CELL 3]
        # We replace the whole block up to # %% [CELL 3: PHASE 2 EVALUATION]
        pattern = r"# %% \[CELL 2: VRAM CLEANUP\].*?# %% \[CELL 3: PHASE 2 EVALUATION\]"
        new_content = re.sub(pattern, new_cell_2, content, flags=re.DOTALL)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated CELL 2 in {script}")
    else:
        print(f"File {path} not found")
