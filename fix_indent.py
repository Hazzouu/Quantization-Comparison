import os

def fix_indent(file_path):
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix MBPP Initialization
    content = content.replace(
        "mbpp_samples = []\n    total_generated_tokens = 0\n",
        "mbpp_samples = []\ntotal_generated_tokens = 0\n"
    )
    
    # Fix HumanEval Initialization
    content = content.replace(
        "he_samples = []\n    total_generated_tokens = 0\n",
        "he_samples = []\ntotal_generated_tokens = 0\n"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed indentation in {file_path}")

fix_indent(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py")
fix_indent(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py")
