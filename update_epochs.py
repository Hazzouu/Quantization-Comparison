import os
import re

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

        # Replace max_steps=100 with num_train_epochs=1
        content = re.sub(r'max_steps\s*=\s*100,?', 'num_train_epochs=1,', content)
        
        # Replace logging_steps=10 with logging_steps=50
        content = re.sub(r'logging_steps\s*=\s*10,?', 'logging_steps=50,', content)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated epochs in {script}")
    else:
        print(f"File {path} not found")
