from pathlib import Path
import pandas as pd

exec(Path(next(Path("/kaggle/input").glob("*/kaggle_support.py"))).read_text(), globals())
REPO = bootstrap_repo()
from rsna_knee.cache import CacheSpec, build_cache_shard
from rsna_knee.config import load_config

ROOT = competition_root()
config = load_config(REPO / "configs/base.yaml")["cache"]
spec = CacheSpec(**{key: config[key] for key in CacheSpec.__dataclass_fields__ if key in config})
studies = pd.read_csv(ROOT / "train.csv")
series = pd.read_csv(ROOT / "train_series.csv")
selected = studies.iloc[(len(studies) + 1) // 2 :].reset_index(drop=True)
print(build_cache_shard(selected, series, ROOT / "train_series", "/kaggle/working/cache_01_of_02", spec))

