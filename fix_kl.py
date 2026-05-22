import os

def fix_compute_kl(file_path):
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The old compute_kl block:
    old_kl = """def compute_kl(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean', log_target=False).item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits)"""

    # The new compute_kl block (PTQ version):
    new_kl = """vocab_size = fp16_logits.size(-1)

def compute_kl(logits_p, logits_q, vocab_size):
    flat_p = logits_p.view(-1, vocab_size)
    flat_q = logits_q.view(-1, vocab_size)
    p_probs = F.softmax(flat_p, dim=-1)
    q_log_probs = F.log_softmax(flat_q, dim=-1)
    kl = F.kl_div(q_log_probs, p_probs, reduction='batchmean').item()
    return kl

kl_div = compute_kl(fp16_logits, qlora_logits, vocab_size)"""

    content = content.replace(old_kl, new_kl)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed compute_kl in {file_path}")

fix_compute_kl(r"c:\Users\youse\Documents\Thesis3\sft_qwen_7b_mbpp.py")
fix_compute_kl(r"c:\Users\youse\Documents\Thesis3\sft_qwen_1_5b_mbpp.py")
