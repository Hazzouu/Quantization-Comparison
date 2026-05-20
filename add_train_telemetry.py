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

telemetry_template = """import time
import json
import threading
import numpy as np
import pynvml

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

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
tracker = PowerTracker()
tracker.start()
start_time = time.time()

trainer.train()

total_time = time.time() - start_time
avg_power = tracker.stop()
tracker.join()
peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

train_loss = None
for log in trainer.state.log_history:
    if 'loss' in log:
        train_loss = log['loss']

trainer.model.save_pretrained("./output/{SCRIPT_NAME}")

metrics = {
    "Total_Time_s": total_time,
    "Peak_VRAM_GB": peak_vram,
    "Avg_Power_W": avg_power,
    "Train_Loss": train_loss
}
with open("{SCRIPT_NAME}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\\n--- Training Telemetry ---")
print(f"Total Training Time: {total_time:.2f} s")
print(f"Average Power:       {avg_power:.2f} W")
print(f"Peak VRAM:           {peak_vram:.2f} GB")
print(f"Train Loss:          {train_loss}\\n")
"""

for script in scripts:
    path = os.path.join(r"c:\Users\youse\Documents\Thesis3", script)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        script_name = script.replace(".py", "")

        # 1. Update SFTConfig to include logging_steps=10 and output to correct dir
        # find sft_config = SFTConfig(...) and replace it entirely
        config_pattern = r'sft_config\s*=\s*SFTConfig\((.*?)\)'
        def config_replacer(match):
            inner = match.group(1)
            # update output_dir
            inner = re.sub(r'output_dir\s*=\s*".*?"', f'output_dir="./output/{script_name}"', inner)
            # add logging_steps if not exists
            if 'logging_steps' not in inner:
                inner = inner.replace('dataset_text_field=', 'logging_steps=10,\n    dataset_text_field=')
            return f"sft_config = SFTConfig({inner})"

        content = re.sub(config_pattern, config_replacer, content, flags=re.DOTALL)

        # 2. Replace trainer.train() and trainer.model.save_pretrained(...)
        train_pattern = r'trainer\.train\(\)\s*\ntrainer\.model\.save_pretrained\(.*?\)'
        
        new_train_block = telemetry_template.replace("{SCRIPT_NAME}", script_name)
        
        if re.search(train_pattern, content):
            content = re.sub(train_pattern, new_train_block, content)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Added training telemetry to {script}")
        else:
            print(f"Train block not found in {script}")
    else:
        print(f"File {path} not found")
