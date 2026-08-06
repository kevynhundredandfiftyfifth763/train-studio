"""Train Studio — dataset loading & preview"""
import json
import os


def normalize_dataset_path(dataset, data_dir):
    """Resolve dataset to a local directory containing train.jsonl/val.jsonl.
    - If dataset is a local path to a .jsonl file -> parent dir, file name as train_file
    - If dataset is a local directory -> use it directly
    - If dataset looks like HF id (has '/') -> return as-is (downloaded by trainer via load_dataset)
    """
    if os.path.isdir(dataset):
        return dataset
    if os.path.isfile(dataset):
        return os.path.dirname(dataset) or "."
    # HF dataset id or something else — pass through
    return dataset


def preview_dataset(dataset, train_file="train.jsonl", val_file="val.jsonl", n=5):
    """Preview first n rows of a dataset (local or HF)."""
    out = {"train": [], "val": [], "info": ""}
    try:
        # local dir?
        if os.path.isdir(dataset):
            for name, fn in [("train", train_file), ("val", val_file)]:
                p = os.path.join(dataset, fn)
                if os.path.exists(p):
                    rows = []
                    with open(p) as f:
                        for i, line in enumerate(f):
                            if i >= n:
                                break
                            try:
                                rows.append(json.loads(line))
                            except Exception:
                                rows.append({"raw": line[:200]})
                    out[name] = rows
            out["info"] = f"Local dir: {dataset}"
            return out
        if os.path.isfile(dataset):
            rows = []
            with open(dataset) as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        rows.append({"raw": line[:200]})
            out["train"] = rows
            out["info"] = f"Local file: {dataset}"
            return out
        # HF dataset id
        out["info"] = f"HF dataset: {dataset} (จะ download ตอนเริ่มเทรน)"
        return out
    except Exception as e:
        out["info"] = f"Error: {e}"
        return out


def format_preview(rows, label):
    """Format preview rows as text."""
    if not rows:
        return f"{label}: (no rows)"
    lines = [f"**{label} — {len(rows)} rows preview:**"]
    for i, r in enumerate(rows):
        text = r.get("text") or r.get("prompt") or r.get("messages") or json.dumps(r, ensure_ascii=False)
        s = str(text)
        if len(s) > 300:
            s = s[:300] + "..."
        lines.append(f"\n--- row {i} ---\n{s}")
    return "\n".join(lines)
