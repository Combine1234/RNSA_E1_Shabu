from pathlib import Path
import hashlib
import json

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.config import load_config
from rsna_knee.models import StudyMILModel
import torch

all_config = load_config(REPO / "configs/base.yaml")
config = all_config["model_b"]
roots = [Path("/kaggle/input/raptor-knee-widedense")]
roots.extend(Path("/kaggle/input/datasets").glob("*/raptor-knee-widedense"))
candidates = sorted({
    path for suffix in ("*.pt", "*.pth", "*.bin")
    for root in roots if root.exists()
    for path in root.glob(f"**/{suffix}") if path.is_file()
})
if not candidates:
    raise FileNotFoundError("Raptor WideDense checkpoint dataset was not mounted")
selected = next((path for path in candidates if "swa" in path.stem.casefold()), candidates[0])

results = []
for path in candidates:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    row = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    try:
        model = StudyMILModel(
            backbone_name=config["backbone"], input_size=int(config["image_size"]),
            backbone_checkpoint=path, last_blocks_trainable=int(config["last_blocks_trainable"]),
            pretrained=False,
        )
        row["compatible"] = True
        if path == selected:
            model.eval()
            size = int(config["image_size"])
            with torch.inference_mode():
                output = model(
                    torch.zeros(1, 6, 3, size, size),
                    torch.ones(1, 6, 3, dtype=torch.bool),
                )
            if tuple(output.shape) != (1, 12):
                raise RuntimeError(f"unexpected synthetic forward shape: {tuple(output.shape)}")
            row["synthetic_forward_shape"] = list(output.shape)
    except Exception as exc:
        row["compatible"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    results.append(row)

dino_config = all_config["model_a"]
dino_size = int(dino_config["image_size"])
dino = StudyMILModel(
    backbone_name=dino_config["backbone"], input_size=dino_size,
    last_blocks_trainable=int(dino_config["last_blocks_trainable"]), pretrained=False,
).eval()
with torch.inference_mode():
    dino_output = dino(
        torch.zeros(1, 6, 3, dino_size, dino_size),
        torch.ones(1, 6, 3, dtype=torch.bool),
    )
if tuple(dino_output.shape) != (1, 12):
    raise RuntimeError(f"unexpected DINO synthetic forward shape: {tuple(dino_output.shape)}")

payload = {
    "dataset": "dreaddevelopment/raptor-knee-widedense",
    "license": "CC0-1.0",
    "backbone": config["backbone"],
    "input_size": int(config["image_size"]),
    "selected_checkpoint": str(selected),
    "candidates": results,
    "selected_checkpoint_compatible": next(
        row["compatible"] for row in results if row["path"] == str(selected)
    ),
    "dino_synthetic_forward_shape": list(dino_output.shape),
}
Path("/kaggle/working/raptor_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
