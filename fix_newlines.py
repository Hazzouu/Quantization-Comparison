import os

scripts = [
    "sft_llama_3_2_gsm8k.py",
    "sft_llama_3_2_mmlu.py",
    "sft_llama_3_8b_gsm8k.py",
    "sft_llama_3_8b_mmlu.py",
    "sft_qwen_1_5b_mbpp.py",
    "sft_qwen_7b_mbpp.py",
    "test_run_pipeline.py"
]

for script in scripts:
    path = os.path.join(r"c:\Users\youse\Documents\Thesis3", script)
    if not os.path.exists(path):
        continue
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix Training Telemetry header
    content = content.replace('print("\n--- Training Telemetry ---")', 'print("\\n--- Training Telemetry ---")')
    
    # Fix Train Loss newline
    content = content.replace('print(f"Train Loss:          {train_loss}\n")', 'print(f"Train Loss:          {train_loss}\\n")')
    
    # Fix VRAM cleared newline
    content = content.replace('print("[*] VRAM cleared and reset successfully.\n")', 'print("[*] VRAM cleared and reset successfully.\\n")')
    
    # Fix CELL headers
    content = content.replace('print("\n" + "="*60)', 'print("\\n" + "="*60)')
    content = content.replace('print("\nCELL ', 'print("\\nCELL ')
    
    # Fix KL Divergence print
    content = content.replace('print(f"\n[EVAL] KL Divergence: {kl_div:.4f}\n")', 'print(f"\\n[EVAL] KL Divergence: {kl_div:.4f}\\n")')
    
    # Fix LoRA SFT Results header (if any got caught)
    content = content.replace('print("\n--- LoRA SFT Results ---")', 'print("\\n--- LoRA SFT Results ---")')
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Fixed formatting in {script}")
