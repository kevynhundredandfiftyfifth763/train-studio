"""Train Studio — training configuration (dataclass + validation)"""
import json
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingConfig:
    # identity
    job_name: str = "train_job"

    # hardware
    gpus: list = field(default_factory=lambda: [0])

    # model / dataset
    model: str = ""
    dataset: str = ""
    hf_token: str = ""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    extra_modules: str = "in_proj_qkv,out_proj,in_proj_z"  # qwen3.5 GDN layers

    # training
    max_seq: int = 4096
    lr: float = 1e-4
    epochs: float = 1.0
    max_steps: int = -1  # -1 = auto from epochs
    batch: int = 2
    grad_accum: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    optim: str = "adamw_8bit"

    # precision
    bf16: bool = True
    fp16: bool = False
    load_in_4bit: bool = False

    # checkpoint
    save_every: int = 50
    save_total_limit: int = 5
    output_dir: str = "/root/train-studio/outputs"

    # VRAM control (per-GPU max memory, GiB)
    custom_vram: bool = False
    vram_per_gpu: str = ""  # e.g. "9,7,7,7" -> GPU0=9GiB, GPU1=7GiB ...

    # dataset split
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"

    def vram_list(self, n_gpus):
        """Return list of per-GPU memory in GiB, or None for auto (9GiB each)."""
        if not self.custom_vram or not self.vram_per_gpu:
            return None
        vals = [x.strip() for x in self.vram_per_gpu.split(",") if x.strip()]
        if not vals:
            return None
        try:
            mems = [float(v) for v in vals]
        except ValueError:
            return None
        if len(mems) < len([g for g in self.gpus]):
            # pad with last value
            mems = mems + [mems[-1]] * (len(self.gpus) - len(mems))
        return mems[:len(self.gpus)]

    def target_list(self):
        t = [x.strip() for x in self.target_modules.split(",") if x.strip()]
        e = [x.strip() for x in self.extra_modules.split(",") if x.strip()]
        return t + e

    def validate(self):
        errs = []
        if not self.model:
            errs.append("Model is required")
        if not self.dataset:
            errs.append("Dataset is required")
        if not self.gpus:
            errs.append("Select at least 1 GPU")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            errs.append("LoRA r/alpha must be > 0")
        if self.max_seq <= 0:
            errs.append("max_seq must be > 0")
        if self.lr <= 0:
            errs.append("lr must be > 0")
        if self.epochs <= 0 and self.max_steps <= 0:
            errs.append("Set epochs or max_steps")
        if self.batch <= 0 or self.grad_accum <= 0:
            errs.append("batch/grad_accum must be > 0")
        return errs

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
