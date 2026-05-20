import os

def add_latency_metrics(file_path):
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # --- MBPP ---
    content = content.replace(
        "mbpp_samples = []\n",
        "mbpp_samples = []\n    total_generated_tokens = 0\n"
    )
    content = content.replace(
        "new_tokens = output[input_length:]\n",
        "new_tokens = output[input_length:]\n            total_generated_tokens += len(new_tokens)\n"
    )
    content = content.replace(
        "elapsed_time_mbpp = time.time() - start_time\n",
        "elapsed_time_mbpp = time.time() - start_time\nt_per_s_mbpp = total_generated_tokens / elapsed_time_mbpp if elapsed_time_mbpp > 0 else 0.0\ns_per_it_mbpp = elapsed_time_mbpp / len(mbpp_data) if len(mbpp_data) > 0 else 0.0\n"
    )
    content = content.replace(
        "print(f\"Time:             {elapsed_time_mbpp:.2f} s\")\n",
        "print(f\"Time:             {elapsed_time_mbpp:.2f} s\")\nprint(f\"Tokens/Sec:       {t_per_s_mbpp:.2f} t/s\")\nprint(f\"Latency s/it:     {s_per_it_mbpp:.4f} s/it\")\n"
    )

    # --- HumanEval ---
    # Be careful not to replace the MBPP new_tokens line again, but since replace() runs globally, 
    # the second new_tokens (in HumanEval) will also get the `total_generated_tokens += len(new_tokens)` appended.
    # Wait, the first replace() for new_tokens replaced ALL instances globally! 
    # That means both MBPP and HumanEval got `total_generated_tokens += len(new_tokens)` added.
    
    # We just need to initialize `total_generated_tokens = 0` for HumanEval:
    content = content.replace(
        "he_samples = []\n",
        "he_samples = []\n    total_generated_tokens = 0\n"
    )
    
    content = content.replace(
        "elapsed_time_he = time.time() - start_time\n",
        "elapsed_time_he = time.time() - start_time\nt_per_s_he = total_generated_tokens / elapsed_time_he if elapsed_time_he > 0 else 0.0\ns_per_it_he = elapsed_time_he / len(he_dataset) if len(he_dataset) > 0 else 0.0\n"
    )
    content = content.replace(
        "print(f\"Time:             {elapsed_time_he:.2f} s\")\n",
        "print(f\"Time:             {elapsed_time_he:.2f} s\")\nprint(f\"Tokens/Sec:       {t_per_s_he:.2f} t/s\")\nprint(f\"Latency s/it:     {s_per_it_he:.4f} s/it\")\n"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Added latency metrics to {file_path}")

add_latency_metrics(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py")
add_latency_metrics(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py")
