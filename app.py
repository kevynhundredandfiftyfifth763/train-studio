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

DEFAULT_MODEL = "/root/models/"
DEFAULT_DATA = "/root/datasets/"
DEFAULT_OUT = "/root/train-studio/outputs"

# ---- UI state persistence (keep user-entered values across tab switches / restarts) ----
UI_STATE = {}
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_state.json")


def load_state():
    global UI_STATE
    try:
        with open(STATE_FILE) as f:
            UI_STATE = json.load(f)
    except Exception:
        UI_STATE = {}
    return UI_STATE


def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(UI_STATE, f, indent=2)
    except Exception:
        pass


def state_val(key, default):
    return UI_STATE.get(key, default)


def auto_job_name(model, dataset):
    """job name = model basename + dataset basename (sanitized).
    ถ้า dataset เป็นไฟล์ (train.jsonl) → ใช้ชื่อ folder แทน"""
    import re
    m = (model or "").rstrip("/").split("/")[-1] if model else ""
    d = (dataset or "").rstrip("/")
    base = os.path.basename(d)
    _exts = (".jsonl", ".json", ".parquet", ".txt", ".csv")
    if os.path.isfile(d) or (base and os.path.splitext(base)[1] in _exts):
        d = os.path.basename(os.path.dirname(d))  # ไฟล์ → ใช้ folder
    else:
        d = base
    name = f"{m}_{d}".strip("_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    return name or "train_job"


def on_model_change(model, dataset):
    """Persist model + auto job name."""
    UI_STATE["model"] = model or ""
    if dataset:
        UI_STATE["job_name"] = auto_job_name(model, dataset)
    save_state()
    return UI_STATE.get("job_name", "train_job")


def on_dataset_change(model, dataset):
    """Persist dataset + auto job name."""
    UI_STATE["dataset"] = dataset or ""
    if model:
        UI_STATE["job_name"] = auto_job_name(model, dataset)
    save_state()
    return UI_STATE.get("job_name", "train_job")


def on_job_name_change(job_name):
    UI_STATE["job_name"] = job_name or ""
    save_state()


def on_hf_token_change(hf_token):
    UI_STATE["hf_token"] = hf_token or ""
    save_state()


# ---- Download (model/dataset) — background thread + log + auto-fill ----
import threading

DOWNLOAD_STATE = {"running": False, "kind": "", "repo": "", "log": "", "target": ""}


def _dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def do_download(kind, repo, hf_token):
    """kind = 'model' | 'dataset'. Download HF repo → /root/models|datasets/<name>"""
    if DOWNLOAD_STATE.get("running"):
        return f"⚠️ กำลัง download `{DOWNLOAD_STATE['repo']}` อยู่ — รอให้เสร็จก่อน"
    if not repo or os.path.exists(repo):
        return f"{kind} เป็น local path อยู่แล้ว หรือช่องว่าง — ไม่ต้อง download"
    base = "/root/models" if kind == "model" else "/root/datasets"
    name = repo.rstrip("/").split("/")[-1]
    target = f"{base}/{name}"
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",
                            f"download_{kind}_{repo.replace('/', '_')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    DOWNLOAD_STATE.update({"running": True, "kind": kind, "repo": repo, "log": log_path, "target": target})
    threading.Thread(target=_dl_worker, args=(kind, repo, hf_token, target, log_path), daemon=True).start()
    return f"⏳ เริ่ม download {kind} `{repo}` → `{target}`\n(กด '🔄 ตรวจสถานะ' เพื่อดูความคืบหน้า)"


def _dl_worker(kind, repo, hf_token, target, log_path):
    try:
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import snapshot_download
        p = snapshot_download(
            repo_id=repo,
            repo_type="dataset" if kind == "dataset" else "model",
            local_dir=target,
        )
        with open(log_path, "w") as f:
            f.write(f"DONE: {p}")
    except Exception as e:
        with open(log_path, "w") as f:
            f.write(f"ERROR: {e}")
    finally:
        DOWNLOAD_STATE["running"] = False


def do_dl_status():
    """สถานะ download — ถ้าเสร็จจะ auto-fill path เข้าช่อง model/dataset."""
    if not DOWNLOAD_STATE.get("log"):
        return "*(ยังไม่มีการ download — ใส่ HF repo id แล้วกด ⬇️)*", gr.update(), gr.update()
    log_path = DOWNLOAD_STATE["log"]
    kind = DOWNLOAD_STATE.get("kind", "")
    done_path = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read().strip()
        if content.startswith("DONE"):
            done_path = content[5:]
        elif content.startswith("ERROR"):
            return f"❌ **Download error:** {content[6:]}", gr.update(), gr.update()
    if DOWNLOAD_STATE.get("running") or (not done_path):
        gb = _dir_size_gb(DOWNLOAD_STATE.get("target", ""))
        return (f"⏳ กำลัง download `{DOWNLOAD_STATE['repo']}`... โหลดไปแล้ว **{gb:.1f} GB** "
                f"(กด 🔄 อีกครั้งเพื่อเช็ค)", gr.update(), gr.update())
    # เสร็จแล้ว — auto-fill path เข้าช่อง
    if kind == "model":
        return f"✅ **Download model เสร็จ:** `{done_path}`\n\nได้เลือกให้อัตโนมัติแล้ว — กด ▶️ Start ได้เลย", gr.update(value=done_path), gr.update()
    return f"✅ **Download dataset เสร็จ:** `{done_path}`\n\nได้เลือกให้อัตโนมัติแล้ว — กด ▶️ Start ได้เลย", gr.update(), gr.update(value=done_path)


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


def inspect_dataset(dataset, train_file, val_file):
    """ตรวจสอบ dataset: ไฟล์ที่พบ + format (text/messages) + rows + ตัวอย่าง"""
    import json as _json
    lines = []
    d = dataset or ""
    if os.path.isfile(d):
        train_file = os.path.basename(d)
        d = os.path.dirname(d)
    if not os.path.isdir(d):
        return f"⚠️ `{d}` ยังไม่ใช่ local folder — ถ้าเป็น HF repo id ระบบจะโหลดตอนเทรน (กด ⬇️ Download ก่อนก็ได้)"
    files = sorted(os.listdir(d))
    lines.append(f"**📁 ไฟล์ใน `{d}`:**")
    lines.append("  " + (", ".join(files[:15]) if files else "(ว่าง)"))
    candidates = [f for f in files if f.endswith((".jsonl", ".json", ".parquet"))]
    if not candidates:
        lines.append("\n❌ ไม่พบไฟล์ .jsonl/.json/.parquet — ตรวจสอบ dataset")
        return "\n".join(lines)
    tf = train_file if train_file in files else (candidates[0] if candidates else None)
    if not tf:
        lines.append("\n❌ ไม่พบ train file")
        return "\n".join(lines)
    path = os.path.join(d, tf)
    try:
        rows = []
        count = 0
        if tf.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(path)
            count = len(df)
            rows = df.head(3).to_dict("records")
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    try:
                        rows.append(_json.loads(line))
                    except Exception:
                        pass
            try:
                count = sum(1 for _ in open(path))
            except Exception:
                pass
        if not rows:
            lines.append(f"\n❌ อ่าน `{tf}` ไม่ได้ หรือไฟล์ว่าง")
            return "\n".join(lines)
        keys = set()
        for r in rows:
            keys.update(r.keys())
        lines.append(f"\n**ไฟล์ train:** `{tf}` ({tf.split('.')[-1]}) — keys: {sorted(keys)}")
        first = rows[0]
        if "text" in first:
            lines.append("✅ format: **text** (ChatML) — ใช้ได้เลย")
        elif "messages" in first:
            lines.append("✅ format: **messages** (OpenAI) — ระบบแปลงเป็น ChatML ให้อัตโนมัติ")
        else:
            lines.append(f"⚠️ ต้องมี key 'text' หรือ 'messages' — keys ที่เจอ: {sorted(keys)}")
        lines.append(f"**จำนวน rows:** {count:,}")
        sample = str(first.get("text") or _json.dumps(first.get("messages", first), ensure_ascii=False))
        lines.append(f"**ตัวอย่าง row แรก:** {sample[:250]}")
        # val check
        vf = val_file if val_file in files else None
        if vf:
            try:
                vpath = os.path.join(d, vf)
                if vf.endswith(".parquet"):
                    vcount = len(pd.read_parquet(vpath))
                else:
                    vcount = sum(1 for _ in open(vpath))
                lines.append(f"**Val file:** `{vf}` ({vcount:,} rows) ✅")
            except Exception:
                pass
        else:
            lines.append(f"ℹ️ ไม่พบ val file (`{val_file}`) — ระบบจะใช้ train 50 ตัวอย่างแทน (ไม่กระทบ)")
    except Exception as e:
        lines.append(f"\n❌ error อ่านไฟล์: {e}")
    return "\n".join(lines)


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


def do_log(n=25):
    """Last N lines (auto-refresh friendly — short enough not to jump)."""
    log_text = trainer.tail_log(n)
    if not log_text.strip():
        return "*(ยังไม่มี log — กด Start Training)*"
    return "```\n" + log_text.rstrip() + "\n```"


def do_full_log():
    """Full log (manual — for reading errors)."""
    log_text = trainer.tail_log(500)
    if not log_text.strip():
        return "*(ยังไม่มี log)*"
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


def list_checkpoints():
    """Scan output dir for checkpoints: outputs/*/checkpoint-* (sorted by job, then step)."""
    import glob
    pat = os.path.join(DEFAULT_OUT, "*", "checkpoint-*")
    ckpts = sorted(glob.glob(pat), key=lambda p: (os.path.basename(os.path.dirname(p)),
                                                  int(os.path.basename(p).split("-")[-1])))
    return ckpts


def checkpoint_choices():
    """Dropdown choices: ตัวเลือก 'ไม่เลือก' แรก + รายการ checkpoint จริง."""
    return [("", "— ไม่เลือก (เริ่มใหม่) —")] + [(c, c) for c in list_checkpoints()]


def do_start(gpus, model, dataset, hf_token, job_name, lora_r, lora_alpha, lora_dropout,
             target_modules, extra_modules, max_seq, lr, epochs, max_steps, batch, grad_accum,
             warmup, wd, scheduler, optim, bf16, fp16, load4bit, save_every, save_limit,
             custom_vram, vram_per_gpu, resume_ck, resume_ckpt, train_file, val_file):
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
        resume=bool(resume_ck), resume_ckpt=str(resume_ckpt or ""),
        train_file=train_file, val_file=val_file,
    )
    # persist UI state (เผื่อ change event ไม่ทัน fire)
    UI_STATE.update({"model": model or "", "dataset": dataset or "",
                     "job_name": job_name or "", "hf_token": hf_token or ""})
    save_state()
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
    if "step" in p and "total" in p and p["total"]:
        pct = p["step"] / p["total"] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"**Progress:** {bar} {p['step']} / {p['total']} (**{pct:.1f}%**)")
    if "loss" in p:
        lines.append(f"**Loss:** {p['loss']}")
    if "lr" in p:
        lines.append(f"**LR:** {p['lr']:.2e}")
    if "resume" in p:
        lines.append(f"**Resume:** {p['resume']}")
    if "error" in p:
        lines.append(f"⚠️ **{p['error']}**")
    return "\n".join(lines)


def do_checkpoint_status():
    """List saved checkpoints with size."""
    ckpts = list_checkpoints()
    if not ckpts:
        return "**Checkpoints:** *(ยังไม่มี — จะ save ทุก N steps หลังเทรนเริ่ม)*"
    lines = ["**Checkpoints (saved):**"]
    for c in ckpts[-12:]:
        job = os.path.basename(os.path.dirname(c))
        step = os.path.basename(c).split("-")[-1]
        size = 0
        for root, _, files in os.walk(c):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        lines.append(f"- `{job}` → checkpoint-{step} ({size/1e6:.0f} MB)")
    lines.append(f"\n*รวม {len(ckpts)} checkpoints — ที่ `/root/train-studio/outputs/`*")
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


def do_merge_status():
    if merger.running:
        return "🔄 **กำลัง merge...** (กด Refresh เพื่อดู log)"
    return "**Status:** idle"


load_state()  # โหลดค่าที่ user เคยบันทึกไว้ (model/dataset/job_name/hf_token)

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
        hf_token = gr.Textbox(label="🤗 HF Token (optional — สำหรับ gated model/dataset)",
                              type="password", value=state_val("hf_token", ""))
        with gr.Row():
            model = gr.Textbox(label="Model (HF repo id หรือ local path — พิมพ์ path แล้วเลือกอัตโนมัติ)",
                               value=state_val("model", DEFAULT_MODEL), scale=4)
            model_browse = gr.Button("📁 Browse", scale=1)
        model_dd = gr.Dropdown(label="เลือก path (พิมพ์ต่อเพื่อค้นหา — folder ลงท้าย /)", choices=[],
                               visible=False, interactive=True)
        with gr.Row():
            dataset = gr.Textbox(label="Dataset (HF repo id หรือ local dir ที่มี train.jsonl/val.jsonl)",
                                 value=state_val("dataset", DEFAULT_DATA), scale=4)
            dataset_browse = gr.Button("📁 Browse", scale=1)
        dataset_dd = gr.Dropdown(label="เลือก path (พิมพ์ต่อเพื่อค้นหา — folder ลงท้าย /)", choices=[],
                                 visible=False, interactive=True)
        train_file = gr.Textbox(label="Train file name", value="train.jsonl")
        val_file = gr.Textbox(label="Val file name", value="val.jsonl")
        job_name = gr.Textbox(label="Job name (ชื่อ output folder — อัตโนมัติจาก model+dataset แก้ได้)",
                              value=state_val("job_name", auto_job_name(
                                  state_val("model", DEFAULT_MODEL), state_val("dataset", DEFAULT_DATA))))
        with gr.Row():
            dl_model_btn = gr.Button("⬇️ Download Model (จาก HF)", variant="secondary")
            dl_dataset_btn = gr.Button("⬇️ Download Dataset (จาก HF)", variant="secondary")
            dl_status_btn = gr.Button("🔄 ตรวจสถานะ download")
        dl_out = gr.Markdown("*(ยังไม่มีการ download — ใส่ HF repo id ในช่อง แล้วกด ⬇️)*")
        dl_model_btn.click(lambda m, t: do_download("model", m, t), inputs=[model, hf_token], outputs=dl_out)
        dl_dataset_btn.click(lambda d, t: do_download("dataset", d, t), inputs=[dataset, hf_token], outputs=dl_out)
        dl_status_btn.click(do_dl_status, outputs=[dl_out, model, dataset])
        with gr.Row():
            inspect_btn = gr.Button("🔍 ตรวจสอบ Dataset (format/rows)", variant="secondary")
            preview_btn = gr.Button("👀 Preview Dataset", variant="secondary")
        preview_out = gr.Markdown()
        inspect_out = gr.Markdown("*(กด 🔍 ตรวจสอบ Dataset — จะบอก format, rows, ตัวอย่าง และผ่าน/ไม่ผ่าน)*")
        preview_btn.click(do_preview, inputs=[model, dataset, hf_token, train_file, val_file], outputs=preview_out)
        inspect_btn.click(inspect_dataset, inputs=[dataset, train_file, val_file], outputs=inspect_out)

        # ---- path autocomplete (file browser) events ----
        model.input(suggest_path, inputs=model, outputs=model_dd)
        model_dd.select(on_path_pick, inputs=model_dd, outputs=[model, model_dd])
        model_browse.click(browse_current, inputs=model, outputs=model_dd)
        dataset.input(suggest_path, inputs=dataset, outputs=dataset_dd)
        dataset_dd.select(on_path_pick, inputs=dataset_dd, outputs=[dataset, dataset_dd])
        dataset_browse.click(browse_current, inputs=dataset, outputs=dataset_dd)

        # ---- persistence + auto job name events ----
        model.change(on_model_change, inputs=[model, dataset], outputs=job_name)
        dataset.change(on_dataset_change, inputs=[model, dataset], outputs=job_name)
        job_name.change(on_job_name_change, inputs=job_name)
        hf_token.change(on_hf_token_change, inputs=hf_token)

    # ---------------- Training Config ----------------
    with gr.Tab("⚙️ Training Config"):
        gpus = gr.CheckboxGroup(label="🎮 เลือก GPU", choices=gpu_choices(), value=gpu_choices())
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
            resume_ck = gr.Checkbox(value=False, label="↩️ Resume จาก checkpoint ล่าสุด (ถ้ามี)")
            start_btn = gr.Button("▶️ Start Training", variant="primary")
            stop_btn = gr.Button("⏹️ Stop", variant="stop")
            refresh_btn = gr.Button("🔄 Refresh")
        with gr.Row():
            resume_ckpt_dd = gr.Dropdown(
                label="เลือก checkpoint ที่จะ resume (ว่าง = เริ่มใหม่ ไม่ resume)",
                choices=checkpoint_choices(), allow_custom_value=True, scale=4)
            scan_ckpt_btn = gr.Button("🔍 Scan", scale=1)
        gpu_vram_out = gr.Markdown("**GPU VRAM (live):** *(กด Refresh หรือรอ auto-update)*")
        ckpt_out = gr.Markdown("**Checkpoints:** *(กด Refresh หรือรอ auto-update)*")
        status_out = gr.Markdown("**Status:** idle")
        log_out = gr.Markdown("*(log จะแสดงอัตโนมัติ — อ่านเต็มได้ที่ Full Log ด้านล่าง)*")
        with gr.Accordion("📜 Full Log (อ่าน error เต็ม)", open=False):
            full_log_btn = gr.Button("🔄 Refresh Full Log")
            full_log_out = gr.Markdown("*(กดปุ่มเพื่อโหลด log เต็ม)*")
        start_btn.click(
            do_start,
            inputs=[gpus, model, dataset, hf_token, job_name, lora_r, lora_alpha, lora_dropout,
                    target_modules, extra_modules, max_seq, lr, epochs, max_steps, batch, grad_accum,
                    warmup, wd, scheduler, optim, bf16, fp16, load4bit, save_every, save_limit,
                    custom_vram, vram_per_gpu, resume_ck, resume_ckpt_dd, train_file, val_file],
            outputs=[status_out, log_out],
        )
        scan_ckpt_btn.click(lambda: gr.update(choices=checkpoint_choices()), outputs=resume_ckpt_dd)
        stop_btn.click(do_stop, outputs=status_out)
        refresh_btn.click(do_status, outputs=status_out)
        refresh_btn.click(do_log, outputs=log_out)
        refresh_btn.click(do_gpu_vram, outputs=gpu_vram_out)
        refresh_btn.click(do_checkpoint_status, outputs=ckpt_out)
        full_log_btn.click(do_full_log, outputs=full_log_out)
        # auto refresh status + GPU VRAM + checkpoints + log (last 25 lines) ทุก 5s
        timer = gr.Timer(5)
        timer.tick(do_status, outputs=status_out)
        timer.tick(do_gpu_vram, outputs=gpu_vram_out)
        timer.tick(do_checkpoint_status, outputs=ckpt_out)
        timer.tick(do_log, outputs=log_out)

    # ---------------- Tools ----------------
    with gr.Tab("🛠️ Tools"):
        gr.Markdown("### 🔧 Merge LoRA → Full Model (BF16, scale + config fix)")
        with gr.Row():
            merge_base = gr.Textbox(label="Base model path (พิมพ์ path แล้วเลือกอัตโนมัติ)", value=DEFAULT_MODEL, scale=4)
            merge_base_browse = gr.Button("📁 Browse", scale=1)
        merge_base_dd = gr.Dropdown(label="เลือก path — folder ลงท้าย /", choices=[], visible=False, interactive=True)
        with gr.Row():
            merge_adapter = gr.Textbox(label="LoRA adapter path (โฟลเดอร์ที่มี adapter_model.safetensors)",
                                       value=os.path.join(DEFAULT_OUT, "train_job/final"), scale=4)
            merge_adapter_browse = gr.Button("📁 Browse", scale=1)
        merge_adapter_dd = gr.Dropdown(label="เลือก path — folder ลงท้าย /", choices=[], visible=False, interactive=True)
        merge_scale = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Merge scale")
        with gr.Row():
            merge_out_dir = gr.Textbox(label="Output dir (ที่เก็บ merged model)", value=os.path.join(DEFAULT_OUT, "merged"), scale=4)
            merge_out_browse = gr.Button("📁 Browse", scale=1)
        merge_out_dd = gr.Dropdown(label="เลือก path — folder ลงท้าย /", choices=[], visible=False, interactive=True)
        merge_gpus = gr.CheckboxGroup(label="GPU สำหรับ merge", choices=gpu_choices(), value=gpu_choices())
        with gr.Row():
            merge_btn = gr.Button("🔧 Start Merge", variant="primary")
            merge_refresh_btn = gr.Button("🔄 Refresh Merge Log")
        merge_status = gr.Markdown("**Status:** idle")
        merge_log = gr.Textbox(label="📜 Merge Log", lines=15)
        merge_btn.click(do_merge, inputs=[merge_base, merge_adapter, merge_scale, merge_out_dir, merge_gpus],
                        outputs=[merge_status, merge_log])
        merge_refresh_btn.click(do_merge_log, outputs=merge_log)
        merge_refresh_btn.click(do_merge_status, outputs=merge_status)

        # ---- path autocomplete events (merge fields) ----
        merge_base.input(suggest_path, inputs=merge_base, outputs=merge_base_dd)
        merge_base_dd.select(on_path_pick, inputs=merge_base_dd, outputs=[merge_base, merge_base_dd])
        merge_base_browse.click(browse_current, inputs=merge_base, outputs=merge_base_dd)
        merge_adapter.input(suggest_path, inputs=merge_adapter, outputs=merge_adapter_dd)
        merge_adapter_dd.select(on_path_pick, inputs=merge_adapter_dd, outputs=[merge_adapter, merge_adapter_dd])
        merge_adapter_browse.click(browse_current, inputs=merge_adapter, outputs=merge_adapter_dd)
        merge_out_dir.input(suggest_path, inputs=merge_out_dir, outputs=merge_out_dd)
        merge_out_dd.select(on_path_pick, inputs=merge_out_dd, outputs=[merge_out_dir, merge_out_dd])
        merge_out_browse.click(browse_current, inputs=merge_out_dir, outputs=merge_out_dd)
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
