"""Train Studio — post-training tools: merge LoRA into full model (scale + mtp-safe config)"""
import os
import subprocess
import json
import sys
from datetime import datetime

PYTHON = os.environ.get("PYTHON") or sys.executable


def generate_merge_script(base_model, adapter, scale, output_dir, gpus):
    """Generate a merge script (BF16 full — NOT 4-bit, safe for linear_attn models).
    Also fixes config: restores text_config + sets mtp_num_hidden_layers=0 unless mtp present.
    """
    return f'''#!/usr/bin/env python3
"""Auto-generated merge by Train Studio — scale {scale}"""
import os, torch, resource, json

resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "{','.join(map(str, gpus))}"

import unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

BASE = r"{base_model}"
ADAPTER = r"{adapter}"
SCALE = {scale}
OUT = r"{output_dir}"

acfg = json.load(open(ADAPTER + "/adapter_config.json"))
scaling = acfg["lora_alpha"] / acfg["r"]
print(f"LoRA r={{acfg['r']}} alpha={{acfg['lora_alpha']}} scaling={{scaling}}, SCALE={{SCALE}}", flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE, max_seq_length=4096, dtype=torch.bfloat16,
    load_in_4bit=False, device_map="auto",
)

lora_model = PeftModel.from_pretrained(model, ADAPTER)

# scale ONLY lora_B (linear in scale — NOT A*B which would square the effect)
for n, p in lora_model.named_parameters():
    if "lora_B" in n:
        p.data.mul_(SCALE)

print("Merging (safe_merge)...", flush=True)
merged = lora_model.merge_and_unload(safe_merge=True)

print("Saving...", flush=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
sft_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
sft_tok.save_pretrained(OUT)

# copy processor files (multimodal models need them for AutoProcessor)
import shutil
for fn in ["preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]:
    src = os.path.join(BASE, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, fn))

# fix config: restore text_config + MTP handling
cfg = json.load(open(OUT + "/config.json"))
orig_cfg = json.load(open(BASE + "/config.json"))
tc = orig_cfg.get("text_config", {{}})
tc_keys = set(tc.keys())
for k in list(cfg.keys()):
    if k in tc_keys and k != "model_type":
        del cfg[k]
cfg["text_config"] = tc
# MTP: keep 0 unless mtp tensors actually exist in adapter/merged
cfg["mtp_num_hidden_layers"] = 0
cfg["mtp_use_dedicated_embeddings"] = False
if "text_config" in cfg:
    cfg["text_config"]["mtp_num_hidden_layers"] = 0
    cfg["text_config"]["mtp_use_dedicated_embeddings"] = False
json.dump(cfg, open(OUT + "/config.json", "w"), indent=2)

print(f"DONE! {{OUT}}", flush=True)
'''


class MergeManager:
    def __init__(self, log_dir="/root/train-studio/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.proc = None
        self.log_path = None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, base_model, adapter, scale, output_dir, gpus):
        if self.running:
            return {"error": "Merge already running"}
        script = os.path.join("/root/train-studio/scripts", "merge_job.py")
        os.makedirs(os.path.dirname(script), exist_ok=True)
        with open(script, "w") as f:
            f.write(generate_merge_script(base_model, adapter, scale, output_dir, gpus))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"merge_{ts}.log")
        logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [PYTHON, "-u", script],
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
        return {"started": True, "pid": self.proc.pid, "log": self.log_path}

    def stop(self):
        import signal
        if self.proc and self.running:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            return {"stopped": True}
        return {"error": "Nothing running"}

    def tail_log(self, n=60):
        if not self.log_path or not os.path.exists(self.log_path):
            return "(no log yet)"
        with open(self.log_path, errors="replace") as f:
            text = f.read()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        return "\n".join(lines[-n:])[-10000:]
