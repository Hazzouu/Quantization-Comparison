import os

def restructure_qwen(file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Make sure the tracker starts clean for MBPP
        if "tracker = PowerTracker()" in line and "BATCH_SIZE = 32" in lines[i+3]:
            new_lines.append(line)
            new_lines.append(lines[i+1]) # tracker.start()
            new_lines.append(lines[i+2]) # start_time = time.time()
            new_lines.append(lines[i+3]) # BATCH_SIZE = 32
            i += 4
            continue
            
        # Intercept the start of HumanEval
        if "# %% [CELL 4: PHASE 2 EVALUATION - HUMANEVAL]" in line or "# --- HumanEval Eval ---" in line:
            new_lines.append("\n# --- Finalize MBPP Metrics ---\n")
            new_lines.append("elapsed_time_mbpp = time.time() - start_time\n")
            new_lines.append("avg_power_mbpp = tracker.stop()\n")
            new_lines.append("tracker.join()\n")
            new_lines.append("peak_vram_mbpp = torch.cuda.max_memory_allocated() / (1024**3)\n")
            new_lines.append("print('\\n--- MBPP SFT Results ---')\n")
            new_lines.append("print(f'Base Pass@1: {base_pass_1:.4f}')\n")
            new_lines.append("print(f'Plus Pass@1: {plus_pass_1:.4f}')\n")
            new_lines.append("print(f'Time: {elapsed_time_mbpp:.2f} s')\n")
            new_lines.append("print(f'Avg Power: {avg_power_mbpp:.2f} W')\n")
            new_lines.append("print(f'Peak VRAM: {peak_vram_mbpp:.2f} GB\\n')\n\n")
            
            new_lines.append("# %% [CELL 4: PHASE 2 EVALUATION - HUMANEVAL]\n")
            new_lines.append("print('\\n' + '='*60)\n")
            new_lines.append("print('CELL 4: PHASE 2 EVALUATION - HUMANEVAL')\n")
            new_lines.append("print('='*60 + '\\n')\n")
            new_lines.append("tracker = PowerTracker()\n")
            new_lines.append("tracker.start()\n")
            new_lines.append("start_time = time.time()\n")
            i += 1
            if "print(\"[*] Loading HumanEval Dataset...\")" not in lines[i]:
                new_lines.append("print(\"[*] Loading HumanEval Dataset...\")\n")
            continue
            
        # Intercept the start of PPL
        if "# %% [CELL 5: PHASE 2 EVALUATION - PPL]" in line or "# --- PPL ---" in line:
            new_lines.append("\n# --- Finalize HumanEval Metrics ---\n")
            new_lines.append("elapsed_time_he = time.time() - start_time\n")
            new_lines.append("avg_power_he = tracker.stop()\n")
            new_lines.append("tracker.join()\n")
            new_lines.append("peak_vram_he = torch.cuda.max_memory_allocated() / (1024**3)\n")
            new_lines.append("print('\\n--- HumanEval SFT Results ---')\n")
            new_lines.append("print(f'Pass@1: {he_pass_at_1:.4f}')\n")
            new_lines.append("print(f'Time: {elapsed_time_he:.2f} s')\n")
            new_lines.append("print(f'Avg Power: {avg_power_he:.2f} W')\n")
            new_lines.append("print(f'Peak VRAM: {peak_vram_he:.2f} GB\\n')\n\n")
            
            new_lines.append("# %% [CELL 5: PHASE 2 EVALUATION - PPL]\n")
            new_lines.append("print('\\n' + '='*60)\n")
            new_lines.append("print('CELL 5: PHASE 2 EVALUATION - PPL')\n")
            new_lines.append("print('='*60 + '\\n')\n")
            i += 1
            if "print(\"[*] Running WikiText-2 PPL Evaluation...\")" not in lines[i]:
                new_lines.append("print(\"[*] Running WikiText-2 PPL Evaluation...\")\n")
            continue
            
        # Remove old consolidated metrics block at the bottom
        if "elapsed_time = time.time() - start_time" in line and "total_items =" in lines[i+1]:
            # Skip the old block
            while i < len(lines) and not "del model, base_model" in lines[i]:
                # But keep the PPL block if we are skipping over it! Wait, PPL was above the consolidated metrics.
                # Actually, in the user's code, PPL block is ABOVE avg_power = tracker.stop()
                pass
            
        if "avg_power = tracker.stop()" in line:
            # We skip the rest of the old consolidated metrics block until we hit the 'del model'
            while i < len(lines) and not "del model, base_model" in lines[i]:
                i += 1
            # Write out PPL metric print before deleting
            new_lines.append("print(f'\\n[EVAL] PPL: {ppl_val:.4f}\\n')\n")
            continue
            
        # If it's a line setting total_items or something we don't need anymore, we can just skip it if it was before PPL.
        # Let's handle the old metrics safely by finding them and removing them.
        if line.startswith("elapsed_time = time.time() - start_time"):
            i += 4
            continue
            
        new_lines.append(line)
        i += 1
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Restructured {file_path}")

restructure_qwen(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py")
restructure_qwen(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py")
