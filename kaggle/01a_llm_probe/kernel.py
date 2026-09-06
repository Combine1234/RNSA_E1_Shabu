from pathlib import Path
import json

import numpy as np
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.config import load_config
from rsna_knee.constants import ID_COLUMN
from rsna_knee.llm_labeler import run_worker

ROOT = competition_root()
config = load_config(REPO / "configs/base.yaml")["open_llm"]
train = pd.read_csv(ROOT / "train.csv")

import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"LLM probe requires T4 x2 runtime; detected {torch.cuda.device_count()} GPU(s)")
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    config["model"], revision=config["revision"], cache_dir="/tmp/rsna-knee-hf"
)
indices = np.unique(np.linspace(0, len(train) - 1, 96, dtype=int))
input_path = Path("/tmp/llm_probe_input.jsonl")
output_path = Path("/tmp/llm_probe_output.jsonl")
with input_path.open("w", encoding="utf-8") as handle:
    for index in indices:
        row = train.iloc[int(index)]
        handle.write(json.dumps({
            "index": int(index), "id": str(row[ID_COLUMN]), "report": str(row.get("Report", ""))
        }, ensure_ascii=False) + "\n")
run_worker(
    input_path, output_path, model_path, int(config["batch_size_per_gpu"]),
    int(config["max_input_tokens"]), int(config["max_new_tokens"]),
)
rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
payload = {
    "model": config["model"], "revision": config["revision"], "rows": len(rows),
    "parse_failures": sum(not row["parse_ok"] for row in rows),
    "responses": rows,
}
Path("/kaggle/working/llm_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print({key: value for key, value in payload.items() if key != "responses"})
