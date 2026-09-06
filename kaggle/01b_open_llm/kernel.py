from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.config import load_config
from rsna_knee.constants import ID_COLUMN, TARGETS

ROOT = competition_root()
config = load_config(REPO / "configs/base.yaml")["open_llm"]
train = pd.read_csv(ROOT / "train.csv")
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"open LLM job requires T4 x2; detected {torch.cuda.device_count()} GPU(s)")

from huggingface_hub import snapshot_download
model_path = snapshot_download(
    config["model"], revision=config["revision"], cache_dir="/tmp/rsna-knee-hf"
)
input_paths = [Path(f"/tmp/llm_input_{device}.jsonl") for device in range(2)]
output_paths = [Path(f"/tmp/llm_output_{device}.jsonl") for device in range(2)]
handles = [path.open("w", encoding="utf-8") for path in input_paths]
try:
    for index, row in train.iterrows():
        handles[index % 2].write(json.dumps({
            "index": int(index), "id": str(row[ID_COLUMN]), "report": str(row.get("Report", ""))
        }, ensure_ascii=False) + "\n")
finally:
    for handle in handles:
        handle.close()

processes = []
for device in range(2):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONPATH"] = str(REPO / "src")
    env["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        sys.executable, "-m", "rsna_knee.llm_labeler",
        "--input", str(input_paths[device]),
        "--output", str(output_paths[device]),
        "--model", str(model_path),
        "--batch-size", str(config["batch_size_per_gpu"]),
        "--max-input-tokens", str(config["max_input_tokens"]),
        "--max-new-tokens", str(config["max_new_tokens"]),
    ]
    processes.append(subprocess.Popen(command, env=env))
for process in processes:
    if process.wait() != 0:
        raise RuntimeError("an open-LLM worker failed")

predictions = []
for path in output_paths:
    predictions.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
predictions.sort(key=lambda row: row["index"])
if len(predictions) != len(train) or any(row["id"] != str(train.iloc[i][ID_COLUMN]) for i, row in enumerate(predictions)):
    raise RuntimeError("open-LLM output IDs do not exactly match train.csv")
result = pd.DataFrame({ID_COLUMN: train[ID_COLUMN].astype(str)})
positive = float(config["positive_probability"])
negative = float(config["negative_probability"])
addressed_confidence = float(config["addressed_confidence"])
for target_index, target in enumerate(TARGETS):
    verdicts = [row["codes"][target_index] for row in predictions]
    result[target] = [positive if value == "Y" else negative if value == "N" else 0.5 for value in verdicts]
    result[f"{target}__conf"] = [addressed_confidence if value in {"Y", "N"} else 0.0 for value in verdicts]
    result[f"{target}__verdict"] = verdicts
result.to_csv("/kaggle/working/open_llm_labels.csv", index=False)
audit = {
    "model": config["model"],
    "requested_revision": config["revision"],
    "resolved_revision": Path(model_path).name,
    "license": config["license"],
    "rows": len(result),
    "parse_failures": int(sum(not row["parse_ok"] for row in predictions)),
    "parse_failure_rate": float(sum(not row["parse_ok"] for row in predictions) / len(predictions)),
}
Path("/kaggle/working/open_llm_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
