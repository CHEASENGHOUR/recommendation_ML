import pandas as pd
import numpy as np

def load_laptop_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # ── Normalise column names ──────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Assign unique ID ────────────────────────────────────────────────────
    df["laptop_id"] = range(len(df))

    # ── Fill missing numeric columns with safe defaults ─────────────────────
    for col, default in [
        ("gpu_vram",      0),
        ("ram_capacity", 8),
        ("ssd",         256),
        ("price",      50000),
        ("cpu_cores",     4),
        ("screen_size",  15),
        ("user_rating",  3.0),
        ("performance_score", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # ── Fill missing string columns ─────────────────────────────────────────
    for col, default in [
        ("name",  "Unknown"),
        ("brand", "Unknown"),
        ("cpu",   "Unknown CPU"),
        ("gpu",   "Integrated"),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)

    # ── Derived feature columns ─────────────────────────────────────────────
    df["gpu_type"]   = df["gpu"].apply(_categorize_gpu)
    df["cpu_tier"]   = df["cpu"].apply(_categorize_cpu)
    df["usage_type"] = df.apply(_determine_usage_type, axis=1)

    print(f"[data_ingest] Loaded {len(df)} laptops | columns: {list(df.columns)}")
    return df

def _categorize_gpu(gpu: str) -> str:
    g = str(gpu).lower()
    if any(x in g for x in ["rtx 4080", "rtx 4090", "rtx 5080"]):
        return "high_end"
    if any(x in g for x in ["rtx 3050", "rtx 3060", "rtx 4050", "rtx 4060", "rtx 4070"]):
        return "mid_range"
    if any(x in g for x in ["gtx", "rx 6500", "rx 6600"]):
        return "entry_gaming"
    if "integrated" in g:
        return "integrated"
    return "other"


def _categorize_cpu(cpu: str) -> str:
    c = str(cpu).lower()
    if any(x in c for x in ["i9", "ryzen 9", "core ultra 9"]):
        return "premium"
    if any(x in c for x in ["i7", "ryzen 7", "core ultra 7"]):
        return "high"
    if any(x in c for x in ["i5", "ryzen 5", "core ultra 5"]):
        return "mid"
    if any(x in c for x in ["i3", "ryzen 3"]):
        return "budget"
    return "entry"


def _determine_usage_type(row: pd.Series) -> str:
    """Safe usage classification — all fields accessed via .get()."""
    gpu_vram     = row.get("gpu_vram", 0)
    price        = row.get("price", 50000)
    ram_capacity = row.get("ram_capacity", 8)
    cpu_tier     = row.get("cpu_tier", "entry")

    if gpu_vram > 0:
        return "gaming"
    if price > 80000:
        return "professional"
    if ram_capacity >= 16 and cpu_tier in ("high", "premium"):
        return "productivity"
    return "everyday"