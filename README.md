# 🚀 Train Studio — SFT LoRA Web UI

Web UI สำหรับฝึก LoRA (SFT) บน GPU ของตัวเอง — เลือก GPU / โมเดล / dataset / ปรับ config ได้อิสระผ่าน browser คล้าย unsloth.ai

อิงจาก workflow ที่พิสูจน์แล้ว: **Unsloth BF16 + HF Trainer + checkpoint/resume** (ใช้เทรน Ornith-1.0-9B และ Agents-A1-4B สำเร็จ)

---

## ✨ Features

| Tab | ความสามารถ |
|---|---|
| 🖥️ Hardware | ตรวจ GPU (ชื่อ/VRAM/util/temp), RAM, disk, python env |
| 📦 Model & Dataset | ใส่ HF token, เลือกโมเดล (HF id / local path), dataset + preview |
| ⚙️ Training Config | เลือก GPU, ปรับ LoRA (r/alpha/dropout/target modules), training (lr/epochs/batch/seq/scheduler), precision (BF16/FP16/QLoRA) |
| 🚀 Train | Start / Stop / Resume + status + log auto-refresh (5s) + checkpoint ทุก N steps |
| 🛠️ Tools | Merge LoRA → Full Model (scale + แก้ config อัตโนมัติ) |

## 🧱 โครงสร้าง

```
train-studio/
├── app.py                  # Gradio UI หลัก
├── requirements.txt
├── backend/
│   ├── hardware.py         # ตรวจ hardware (nvidia-smi + /proc)
│   ├── config.py           # TrainingConfig dataclass + validation
│   ├── dataset.py          # load + preview dataset (local/HF)
│   ├── trainer.py          # generate train script + subprocess manager + progress parser
│   └── merge.py            # merge LoRA (BF16, scale, config fix)
├── scripts/                # train/merge scripts ที่ generate อัตโนมัติ
├── logs/                   # training logs
├── outputs/                # checkpoints + LoRA + merged models
└── venv/                   # virtualenv (gradio)
```

## 📦 การติดตั้ง

### ⚡ One-click install (เร็วสุด)

```bash
curl -sSL https://raw.githubusercontent.com/nanofatdog/train-studio/master/install.sh | bash
```

- ติดตั้งที่ `~/train-studio` อัตโนมัติ (ตรวจ Python/GPU/CUDA → เลือก torch ให้เอง)
- ตัวเลือก: `APP_DIR=/path RUN_AFTER=1 curl -sSL ... | bash` (ติดตั้งที่อื่น + รันเลย)

### หรือ manual

```bash
# (เครื่องที่มี GPU / Docker ที่มี nvidia runtime)
git clone https://github.com/nanofatdog/train-studio.git
cd train-studio

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

### 2. ติดตั้ง training stack (ถ้ายังไม่มี)

```bash
# torch ตาม CUDA ของเครื่อง (ตัวอย่าง CUDA 13.0)
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130

# unsloth (สำหรับโหลดโมเดลเร็ว + BF16 patch)
./venv/bin/pip install unsloth
# หรือตาม https://github.com/unslothai/unsloth

# ส่วนที่เหลือใน requirements.txt
./venv/bin/pip install -r requirements.txt
```

> **หมายเหตุ:** ถ้ามี venv ของงานเทรนเดิมอยู่แล้ว (เช่น `/root/agents_sft` ที่มี unsloth+transformers ครบ) — ตั้ง env `PYTHON` ชี้ไป python ของ venv นั้น ตอนรัน app เพื่อให้ subprocess ใช้ unsloth ได้:
> ```bash
> PYTHON=/root/agents_sft/bin/python3 ./venv/bin/python app.py
> ```

## ▶️ การรัน

```bash
cd train-studio
./venv/bin/python app.py
# หรือชี้ PYTHON ไป venv ที่มี unsloth (ดูข้างบน)
```

เปิด browser: `http://localhost:7860`

### การเข้าถึงจากเครื่องอื่น (Docker / remote)

ถ้า UI รันใน container/remote ที่ port 7860 ไม่ได้เปิด:

```bash
# SSH tunnel
ssh -L 7860:localhost:7860 -p <port> user@<host>
# แล้วเปิด http://localhost:7860
```

หรือเพิ่ม port mapping ตอนสร้าง container: `-p 7860:7860`

## 🎮 วิธีใช้งาน

1. **Hardware tab** — กด 🔍 Refresh ดู GPU ที่มี
2. **Model & Dataset tab** — ใส่ HF token (ถ้า gated), เลือกโมเดล + dataset, กด 👀 Preview
   - โมเดล: HF repo id (`Qwen/Qwen3.5-9B`) หรือ local path (`/root/models/Ornith-1.0-9B`)
   - dataset: HF repo id หรือ local dir ที่มี `train.jsonl` / `val.jsonl` (format: `{"text": "<|im_start|>...<|im_end|>"}`)
3. **Training Config tab** — เลือก GPU + ปรับพารามิเตอร์
4. **Train tab** — กด ▶️ Start แล้วดู status/log อัตโนมัติ
   - checkpoint ทุก N steps → resume อัตโนมัติถ้าไฟดับ/หยุด
5. **Tools tab** — merge LoRA → full model (เลือก scale, กด 🔧 Start Merge)

## ⚠️ ข้อควรระวัง (จากประสบการณ์จริง)

### 🔧 แก้ปัญหา "No module named torch" (ตอนกด Start Training)

สาเหตุ: subprocess python (ที่ใช้รันสคริปต์เทรน) ไม่มี torch/unsloth — UI ตรวจได้ที่ **Hardware tab → "Subprocess check"**

วิธีแก้ (เลือก 1):
```bash
# 1. ติดตั้งลง venv ของ train-studio (แนะนำ — ทำครั้งเดียว)
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
./venv/bin/pip install unsloth

# 2. หรือรัน app ด้วย PYTHON env ชี้ไป venv ที่มี torch อยู่แล้ว
PYTHON=/path/to/venv-with-torch/bin/python ./venv/bin/python app.py
```

> install.sh จัดการข้อนี้ให้อัตโนมัติ (ตรวจ CUDA → ติดตั้ง torch + unsloth)

- **qwen3_5 ต้อง BF16 เท่านั้น** — GDN (linear_attn) layers ทำ NaN grad norms ใน FP16
- **qwen3_5 target modules** ต้องเพิ่ม `in_proj_qkv, out_proj, in_proj_z` (นอกจาก q/k/v/o/gate/up/down) — ไม่งั้น 24/32 layers ได้ LoRA แค่ที่ MLP
- **Multimodal tokenizer** (Qwen3.5/Qwen3VL) — ห้ามเรียก `tokenizer(text)` ตรงๆ (มันคิดว่า text เป็น image URL) — ใช้ `tokenizer.tokenizer` (raw) — ตัว generate script จัดการให้แล้ว
- **Merge scale ต้องคูณแค่ `lora_B`** — ถ้าคูณทั้ง A+B จะได้ effect เป็น S² (0.7 จริงๆ = 0.49)
- **MTP config** — ถ้าโมเดลไม่มี mtp tensors จริง ต้องตั้ง `mtp_num_hidden_layers=0` + convert GGUF ด้วย `--no-mtp` (ไม่งั้น block_count เกิน → load error `blk.NN not found`)
- **VRAM** — 9B BF16 บน 4×3060: ใช้ batch=1 + grad_accum=4 + max_memory 7GiB/GPU (batch=2 OOM)

## 🛠️ GGUF Convert (หลัง merge)

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py python3 /path/to/llama.cpp/convert_hf_to_gguf.py <merged_dir> \
  --outfile model.gguf --outtype f16 --no-mtp   # ถ้าไม่มี mtp tensors
# ถ้ามี mtp: ตัด --no-mtp ออก (convert ธรรมดา → block_count = layers+1)
```

> ต้องใช้ llama.cpp version ที่รองรับ qwen3_5 (master หลัง refactor `conversion/qwen.py`)

## 📄 License

MIT
