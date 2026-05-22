import os

def fix_prompts(file_path):
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix MBPP Prompts
    content = content.replace(
        'prompts = [item[1]["prompt"] for item in batch]',
        'prompts = [f"Prompt: Complete the following Python code:\\n{item[1][\'prompt\']}\\n\\nCode:\\n{item[1][\'prompt\']}" for item in batch]'
    )
    
    # Fix HumanEval Prompts
    content = content.replace(
        'prompts = batch["prompt"]',
        'prompts = [f"Prompt: Complete the following Python code:\\n{p}\\n\\nCode:\\n{p}" for p in batch["prompt"]]'
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed prompts in {file_path}")

fix_prompts(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py")
fix_prompts(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py")
