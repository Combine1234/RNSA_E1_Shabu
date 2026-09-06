from pathlib import Path
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.blend import fit_blend
from rsna_knee.config import load_config
from rsna_knee.constants import ID_COLUMN

families = {}
for family in ("dino", "coatnet"):
    slug = "rsna-knee-03-train-dino" if family == "dino" else "rsna-knee-04-train-coatnet"
    paths = files([f"/kaggle/input/**/{slug}/oof_fold_*.csv"], f"{family} OOF folds")
    if len(paths) != 5:
        raise FileNotFoundError(f"expected five {family} OOF folds, found {paths}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).sort_values(ID_COLUMN)
    out = Path(f"/kaggle/working/oof_{family}.csv")
    frame.to_csv(out, index=False)
    families[family] = out
config = load_config(REPO / "configs/base.yaml")["blend"]
print(fit_blend(
    [families["dino"], families["coatnet"]],
    "/kaggle/working/blend_weights.json",
    step=float(config["grid_step"]),
    minimum_gain=float(config["minimum_pseudo_gain"]),
    maximum_gold_drop=float(config["maximum_gold_drop"]),
))
