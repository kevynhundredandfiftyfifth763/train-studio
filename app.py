#!/usr/bin/env python3
"""Train Studio — Gradio Web UI for SFT LoRA training (unsloth-style)
Run: python3 app.py  -> http://0.0.0.0:7860
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from backend.hardware import hardware_summary, get_gpus
from backend.config import TrainingConfig
from backend.dataset import preview_dataset, format_preview
from backend.trainer import TrainerManager, write_train_script
from backend.merge import MergeManager

trainer = TrainerManager()
merger = MergeManager()

DEFAULT_MODEL = "/root/models/Ornith-1.0-9B"
DEFAULT_DATA = "/root/datasets/ornith_fable"
DEFAULT_OUT = "/root/train-studio/outputs"


def gpu_choices():
    gpus = get_gpus()
    if not gpus or "error" in gpus[0]:
        return [0]
    return [g["index"] for g in gpus]


def do_refresh_hw():
    return hardware_summary()


def do_preview(model, dataset, hf_token, train_file, val_file):
    # dataset may be HF id — resolve local if possible
    info_lines = [f"**Model:** {model or '(none)'}"]
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        info_lines.append("**HF Token:** set ✅")
    else:
        info_lines.append("**HF Token:** (none)")
    pv = preview_dataset(dataset, train_file, val_file)
    info_lines.append(pv.get("info", ""))
    txt = "\n".join(info_lines)
    txt += "\n\n" + format_preview(pv.get("train", []), "TRAIN")
    txt += "\n\n" + format_preview(pv.get("val", []), "VAL")
    return txt


# ---- Path autocomplete (file browser for local paths) ----
def suggest_path(prefix):
    """List files/dirs matching a path prefix (shell-like completion)."""
    if not prefix or not prefix.startswith("/"):
        return gr.update(visible=False, choices=[])
    if prefix.endswith("/"):
        base = prefix
    else:
        base = os.path.dirname(prefix) + "/"
        if base == "//":
            base = "/"
    try:
        entries = sorted(os.listdir(base))
    except Exception:
        return gr.update(visible=False, choices=[])
    choices = []
    for e in entries:
        if e.startswith("."):
            continue
        full = os.path.join(base, e)
        choices.append(full + "/" if os.path.isdir(full) else full)
    choices = [c for c in choices if c.startswith(prefix)]
    return gr.update(visible=bool(choices), choices=choices[:60])


def on_path_pick(choice):
    """When user picks a suggestion: folder -> continue browsing, file -> done."""
    if not choice:
        return gr.update(), gr.update(visible=False)
    if choice.endswith("/"):
        return choice, suggest_path(choice)
    return choice, gr.update(visible=False)


def browse_current(path):
    """Browse button: show contents of current path (or /root)."""
    p = path or "/root"
    if not p.endswith("/"):
        p = p + "/"
    return suggest_path(p)


def do_log():
    log_text = trainer.tail_log(50)
    if not log_text.strip():
        return "*(ยังไม่มี log — กด Start Training)*"
    return "```\n" + log_text.rstrip() + "\n```"


def do_gpu_vram():
    """Live per-GPU VRAM usage."""
    gpus = get_gpus()
    lines = ["**GPU VRAM (live):**"]
    for g in gpus:
        if "error" in g:
            continue
        used = g["memory_used_mb"] / 1024
        total = g["memory_total_mb"] / 1024
        pct = g["memory_used_mb"] / g["memory_total_mb"] * 100
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        lines.append(f"- GPU {g['index']}: {bar} **{used:.1f} / {total:.1f} GB** ({pct:.0f}%)")
    return "\n".join(lines)


def do_start(gpus, model, dataset, hf_token, job_name, lora_r, lora_alpha, lora_dropout,
             target_modules, extra_modules, max_seq, lr, epochs, max_steps, batch, grad_accum,
             warmup, wd, scheduler, optim, bf16, fp16, load4bit, save_every, save_limit,
             custom_vram, vram_per_gpu, resume_ck, train_file, val_file):
    cfg = TrainingConfig(
        job_name=job_name or "train_job",
        gpus=[int(g) for g in gpus] if gpus else [0],
        model=model, dataset=dataset, hf_token=hf_token,
        lora_r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
        target_modules=target_modules, extra_modules=extra_modules,
        max_seq=int(max_seq), lr=float(lr), epochs=float(epochs), max_steps=int(max_steps),
        batch=int(batch), grad_accum=int(grad_accum),
        warmup_ratio=float(warmup), weight_decay=float(wd),
        scheduler=scheduler, optim=optim,
        bf16=bool(bf16), fp16=bool(fp16), load_in_4bit=bool(load4bit),
        save_every=int(save_every), save_total_limit=int(save_limit),
        output_dir=DEFAULT_OUT,
        custom_vram=bool(custom_vram), vram_per_gpu=str(vram_per_gpu or ""),
        resume=bool(resume_ck),
        train_file=train_file, val_file=val_file,
    )
    errs = cfg.validate()
    if errs:
        return "\n".join("❌ " + e for e in errs), ""
    res = trainer.start(cfg)
    if "error" in res:
        return "❌ " + res["error"], ""
    return f"✅ Started! PID {res['pid']}\nLog: {res['log']}\nScript: {res['script']}", trainer.tail_log(40)


def do_stop():
    res = trainer.stop()
    return str(res)


def do_status():
    p = trainer.progress()
    lines = [f"**Status:** {p.get('status')}"]
    if "step" in p and "total" in p:
        lines.append(f"**Step:** {p['step']} / {p['total']}")
    if "loss" in p:
        lines.append(f"**Loss:** {p['loss']}")
    if "lr" in p:
        lines.append(f"**LR:** {p['lr']:.2e}")
    if "resume" in p:
        lines.append(f"**Resume:** {p['resume']}")
    if "error" in p:
        lines.append(f"⚠️ **{p['error']}**")
    return "\n".join(lines)


def do_log():
    return trainer.tail_log(100)


def do_merge(base_model, adapter, scale, output_dir, gpus):
    if not base_model or not adapter:
        return "❌ Need base model + adapter paths", ""
    out = output_dir or os.path.join(DEFAULT_OUT, "merged")
    res = merger.start(base_model, adapter, float(scale), out, [int(g) for g in gpus] if gpus else [0])
    if "error" in res:
        return "❌ " + res["error"], ""
    return f"✅ Merge started! PID {res['pid']}", merger.tail_log(40)


def do_merge_log():
    return merger.tail_log(60)


with gr.Blocks(title="Train Studio") as demo:
    gr.Markdown("# 🚀 Train Studio — SFT LoRA Web UI")
    gr.Markdown("เทรน LoRA ผ่านเว็บ — เลือก GPU/โมเดล/dataset/ปรับ config ได้อิสระ (อิง proven workflow: Unsloth BF16 + HF Trainer)")

    # ---------------- Hardware ----------------
    with gr.Tab("🖥️ Hardware"):
        hw_btn = gr.Button("🔍 Refresh Hardware", variant="primary")
        hw_out = gr.Markdown()
        hw_btn.click(do_refresh_hw, outputs=hw_out)
        demo.load(do_refresh_hw, outputs=hw_out)

    # ---------------- Model & Dataset ----------------
    with gr.Tab("📦 Model & Dataset"):
        hf_token = gr.Textbox(label="🤗 HF Token (optional — สำหรับ gated model/dataset)", type="password")
        with gr.Row():
            model = gr.Textbox(label="Model (HF repo id หรือ local path — พิมพ์ path แล้วเลือกอัตโนมัติ)",
                               value=DEFAULT_MODEL, scale=4)
            model_browse = gr.Button("📁 Browse", scale=1)
        model_dd = gr.Dropdown(label="เลือก path (พิมพ์ต่อเพื่อค้นหา — folder ลงท้าย /)", choices=[],
                               visible=False, interactive=True)
        with gr.Row():
            dataset = gr.Textbox(label="Dataset (HF repo id หรือ local dir ที่มี train.jsonl/val.jsonl)",
                                 value=DEFAULT_DATA, scale=4)
            dataset_browse = gr.Button("📁 Browse", scale=1)
        dataset_dd = gr.Dropdown(label="เลือก path (พิมพ์ต่อเพื่อค้นหา — folder ลงท้าย /)", choices=[],
                                 visible=False, interactive=True)
        train_file = gr.Textbox(label="Train file name", value="train.jsonl")
        val_file = gr.Textbox(label="Val file name", value="val.jsonl")
        preview_btn = gr.Button("👀 Preview Dataset", variant="secondary")
        preview_out = gr.Markdown()
        preview_btn.click(do_preview, inputs=[model, dataset, hf_token, train_file, val_file], outputs=preview_out)

        # ---- path autocomplete (file browser) events ----
        model.input(suggest_path, inputs=model, outputs=model_dd)
        model_dd.select(on_path_pick, inputs=model_dd, outputs=[model, model_dd])
        model_browse.click(browse_current, inputs=model, outputs=model_dd)
        dataset.input(suggest_path, inputs=dataset, outputs=dataset_dd)
        dataset_dd.select(on_path_pick, inputs=dataset_dd, outputs=[dataset, dataset_dd])
        dataset_browse.click(browse_current, inputs=dataset, outputs=dataset_dd)

    # ---------------- Training Config ----------------
    with gr.Tab("⚙️ Training Config"):
        gpus = gr.CheckboxGroup(label="🎮 เลือก GPU", choices=gpu_choices(), value=gpu_choices())
        job_name = gr.Textbox(label="Job name (ชื่อ output folder)", value="train_job")
        with gr.Row():
            lora_r = gr.Slider(1, 256, value=64, step=1, label="LoRA r")
            lora_alpha = gr.Slider(1, 512, value=128, step=1, label="LoRA alpha")
            lora_dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="LoRA dropout")
        target_modules = gr.Textbox(label="Target modules (คั่นด้วย ,)", value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        extra_modules = gr.Textbox(label="Extra modules (qwen3.5 GDN layers)", value="in_proj_qkv,out_proj,in_proj_z")
        with gr.Row():
            max_seq = gr.Slider(512, 8192, value=4096, step=512, label="Max seq length")
            lr = gr.Number(value=1e-4, label="Learning rate")
            epochs = gr.Number(value=1.0, label="Epochs")
            max_steps = gr.Number(value=-1, label="Max steps (-1 = auto)")
        with gr.Row():
            batch = gr.Number(value=2, label="Batch size")
            grad_accum = gr.Number(value=2, label="Grad accumulation")
            warmup = gr.Number(value=0.03, label="Warmup ratio")
            wd = gr.Number(value=0.01, label="Weight decay")
        with gr.Row():
            scheduler = gr.Dropdown(["cosine", "linear", "constant"], value="cosine", label="LR scheduler")
            optim = gr.Dropdown(["adamw_8bit", "adamw_torch"], value="adamw_8bit", label="Optimizer")
            save_every = gr.Number(value=50, label="Checkpoint every N steps")
            save_limit = gr.Number(value=5, label="Keep last N checkpoints")
        with gr.Row():
            bf16 = gr.Checkbox(value=True, label="BF16 (แนะนำสำหรับ qwen3.5 — GDN layers)")
            fp16 = gr.Checkbox(value=False, label="FP16 (ระวัง NaN บน qwen3.5)")
            load4bit = gr.Checkbox(value=False, label="QLoRA 4-bit (ประหยัด VRAM)")
        with gr.Row():
            custom_vram = gr.Checkbox(value=False, label="ปรับ VRAM ต่อ GPU ด้วยมือ (default: 9GiB/GPU)")
            vram_per_gpu = gr.Textbox(label="VRAM per GPU (GiB, คั่นด้วย , เช่น 9,7,7,7 — GPU0=9, GPU1=7...)",
                                      value="9,9,9,9", interactive=True)

    # ---------------- Train ----------------
    with gr.Tab("🚀 Train"):
        with gr.Row():
            resume_ck = gr.Checkbox(value=True, label="↩️ Resume จาก checkpoint ล่าสุด (ถ้ามี)")
            start_btn = gr.Button("▶️ Start Training", variant="primary")
            stop_btn = gr.Button("⏹️ Stop", variant="stop")
            refresh_btn = gr.Button("🔄 Refresh")
        gpu_vram_out = gr.Markdown("**GPU VRAM (live):** *(กด Refresh หรือรอ auto-update)*")
        status_out = gr.Markdown("**Status:** idle")
        log_out = gr.Markdown("*(log จะแสดงเมื่อกด Refresh หรือ Start — อัปเดตแบบ manual เพื่อไม่ให้เด้งขึ้นบน)*")
        start_btn.click(
            do_start,
            inputs=[gpus, model, dataset, hf_token, job_name, lora_r, lora_alpha, lora_dropout,
                    target_modules, extra_modules, max_seq, lr, epochs, max_steps, batch, grad_accum,
                    warmup, wd, scheduler, optim, bf16, fp16, load4bit, save_every, save_limit,
                    custom_vram, vram_per_gpu, resume_ck, train_file, val_file],
            outputs=[status_out, log_out],
        )
        stop_btn.click(do_stop, outputs=status_out)
        refresh_btn.click(do_status, outputs=status_out)
        refresh_btn.click(do_log, outputs=log_out)
        refresh_btn.click(do_gpu_vram, outputs=gpu_vram_out)
        # auto refresh status + GPU VRAM ทุก 5s (log แบบ manual — กันเด้ง)
        timer = gr.Timer(5)
        timer.tick(do_status, outputs=status_out)
        timer.tick(do_gpu_vram, outputs=gpu_vram_out)

    # ---------------- Tools ----------------
    with gr.Tab("🛠️ Tools"):
        gr.Markdown("### 🔧 Merge LoRA → Full Model (BF16, scale + config fix)")
        merge_base = gr.Textbox(label="Base model path", value=DEFAULT_MODEL)
        merge_adapter = gr.Textbox(label="LoRA adapter path", value=os.path.join(DEFAULT_OUT, "train_job/final"))
        merge_scale = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Merge scale")
        merge_out_dir = gr.Textbox(label="Output dir", value=os.path.join(DEFAULT_OUT, "merged"))
        merge_gpus = gr.CheckboxGroup(label="GPU สำหรับ merge", choices=gpu_choices(), value=gpu_choices())
        merge_btn = gr.Button("🔧 Start Merge", variant="primary")
        merge_status = gr.Markdown("**Status:** idle")
        merge_log = gr.Textbox(label="📜 Merge Log", lines=15)
        merge_btn.click(do_merge, inputs=[merge_base, merge_adapter, merge_scale, merge_out_dir, merge_gpus],
                        outputs=[merge_status, merge_log])
        gr.Markdown("### 📄 GGUF Convert (รันเองใน terminal)")
        gr.Markdown(
            "```bash\n"
            "PYTHONPATH=/root/llama.cpp/gguf-py python3 /root/llama.cpp/convert_hf_to_gguf.py <merged_dir> \\\n"
            "  --outfile model.gguf --outtype f16 --no-mtp   # ไม่มี MTP\n"
            "# หรือถ้าโมเดลมี mtp tensors: ตัด --no-mtp ออก (convert ธรรมดา)\n"
            "```"
        )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_error=True,
                        theme=gr.themes.Soft())
