"""Train Studio — hardware detection"""
import os
import shutil
import subprocess


def get_gpus():
    """List NVIDIA GPUs via nvidia-smi (name, VRAM, util, temp)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in out.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization": int(parts[4]),
                    "temperature": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def get_ram():
    """RAM total/used in GB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k] = int(v.strip().split()[0])  # kB
        total = mem.get("MemTotal", 0) / 1024 / 1024
        avail = mem.get("MemAvailable", mem.get("MemFree", 0)) / 1024 / 1024
        return {"total_gb": round(total, 1), "available_gb": round(avail, 1)}
    except Exception as e:
        return {"error": str(e)}


def get_disk(path="/root"):
    """Disk usage for a path."""
    try:
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
    except Exception as e:
        return {"error": str(e)}


def get_python_env():
    """Python / torch / transformers versions."""
    import importlib
    info = {"python": os.sys.version.split()[0]}
    for pkg in ["torch", "transformers", "peft", "unsloth", "gradio"]:
        try:
            m = importlib.import_module(pkg)
            info[pkg] = getattr(m, "__version__", "?")
        except Exception:
            info[pkg] = "not installed"
    try:
        import torch
        info["cuda_available"] = str(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = "n/a"
    return info


def hardware_summary():
    """Full summary used by the UI."""
    gpus = get_gpus()
    ram = get_ram()
    disk = get_disk()
    env = get_python_env()
    lines = []
    lines.append(f"**GPUs ({len(gpus)}):**")
    for g in gpus:
        if "error" in g:
            lines.append(f"- ❌ {g['error']}")
        else:
            free = g["memory_total_mb"] - g["memory_used_mb"]
            lines.append(
                f"- GPU {g['index']}: {g['name']} | {g['memory_used_mb']}/{g['memory_total_mb']} MiB "
                f"(free {free} MiB) | util {g['utilization']}% | {g['temperature']}°C"
            )
    lines.append(f"\n**RAM:** {ram.get('total_gb', '?')} GB total / {ram.get('available_gb', '?')} GB available")
    lines.append(f"**Disk ({disk.get('total_gb', '?')} GB):** {disk.get('free_gb', '?')} GB free")
    lines.append(f"\n**Env:** python {env.get('python')} | torch {env.get('torch')} (cuda={env.get('cuda_available')}) | "
                 f"transformers {env.get('transformers')} | peft {env.get('peft')} | unsloth {env.get('unsloth')}")
    return "\n".join(lines)
