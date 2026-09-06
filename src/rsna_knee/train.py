from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ID_COLUMN, TARGETS
from .dataset import BlockShuffleSampler, CachedStudyDataset
from .metrics import macro_auc
from .models import StudyMILModel, module_state_sha256


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def masked_soft_bce(logits, targets, weights, positive_weight=None):
    import torch

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=positive_weight
    )
    weighted = loss * weights
    return weighted.sum() / weights.sum().clamp_min(1.0)


class ModelEMA:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def update(self, model) -> None:
        current = model.state_dict()
        for key, value in current.items():
            detached = value.detach()
            if not detached.is_floating_point():
                self.state[key].copy_(detached)
            else:
                self.state[key].mul_(self.decay).add_(detached, alpha=1.0 - self.decay)


def _predict(model, loader, device: str):
    import torch

    ids: list[str] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    gold: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(images, valid)
            ids.extend(batch["id"])
            predictions.append(torch.sigmoid(logits).float().cpu().numpy())
            truths.append(batch["targets"].numpy())
            weights.append(batch["weights"].numpy())
            gold.append(batch["gold"].numpy())
    return ids, np.concatenate(predictions), np.concatenate(truths), np.concatenate(weights), np.concatenate(gold)


def train_fold(
    labels_path: str | Path,
    cache_paths: list[str | Path],
    output_dir: str | Path,
    model_config: dict[str, Any],
    fold: int,
    seed: int = 20260905,
    gold_weight: float = 4.0,
    num_workers: int = 2,
    backbone_checkpoint: str | Path | None = None,
    backbone_checkpoint_license: str | None = None,
) -> dict[str, Any]:
    import pandas as pd
    import torch

    seed_everything(seed + fold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(labels_path) if str(labels_path).endswith((".parquet", ".pq")) else pd.read_csv(labels_path)
    if "fold" not in labels:
        raise ValueError("labels file must contain a fold column")
    train_indices = np.flatnonzero(labels["fold"].to_numpy() != fold)
    valid_indices = np.flatnonzero(labels["fold"].to_numpy() == fold)
    if not len(train_indices) or not len(valid_indices):
        raise ValueError(f"fold {fold} produced an empty train or validation partition")

    train_data = CachedStudyDataset(
        labels, cache_paths, train_indices, training=True, seed=seed + fold, gold_weight=gold_weight
    )
    valid_data = CachedStudyDataset(labels, cache_paths, valid_indices, training=False, gold_weight=gold_weight)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=int(model_config["batch_size"]),
        shuffle=False,
        sampler=BlockShuffleSampler(
            len(train_data),
            block_size=int(model_config.get("io_block_size", 64)),
            seed=seed + fold,
        ),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=max(1, int(model_config["batch_size"]) * 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("training jobs require a Kaggle GPU runtime")
    device = "cuda:0"
    persisted_model_config = {
        "backbone_name": model_config["backbone"],
        "input_size": int(model_config["image_size"]),
        "n_targets": len(TARGETS),
        "backbone_checkpoint": None,
        "last_blocks_trainable": int(model_config.get("last_blocks_trainable", -1)),
        "pretrained": False,
    }
    initialization = {
        "requested_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
        "requested_checkpoint_license": backbone_checkpoint_license,
        "actual_source": "timm_pretrained" if backbone_checkpoint is None else "external_checkpoint",
        "actual_license": "Apache-2.0" if backbone_checkpoint is None else backbone_checkpoint_license,
    }
    try:
        model = StudyMILModel(
            **{
                **persisted_model_config,
                "backbone_checkpoint": backbone_checkpoint,
                "pretrained": backbone_checkpoint is None,
            }
        )
    except (RuntimeError, ValueError) as exc:
        if backbone_checkpoint is None:
            raise
        print(f"WARNING incompatible external checkpoint ({exc}); falling back to pinned timm initialization")
        initialization["actual_source"] = "timm_pretrained_fallback"
        initialization["actual_license"] = "Apache-2.0"
        initialization["fallback_reason"] = str(exc)
        model = StudyMILModel(**{**persisted_model_config, "pretrained": True})
    initialization["encoder_sha256"] = module_state_sha256(model.encoder)
    model = model.to(device)
    target_values = labels.loc[train_indices, list(TARGETS)].to_numpy(dtype=np.float32)
    confidence_values = np.column_stack([
        labels.loc[train_indices, f"{target}__confidence"].to_numpy(dtype=np.float32) for target in TARGETS
    ])
    gold_values = np.column_stack([
        labels.loc[train_indices, f"{target}__gold"].to_numpy(dtype=bool) for target in TARGETS
    ])
    training_weights = confidence_values * np.where(gold_values, gold_weight, 1.0)
    positives = (training_weights * target_values).sum(axis=0)
    negatives = (training_weights * (1.0 - target_values)).sum(axis=0)
    positive_weight = torch.tensor(
        np.clip(negatives / np.maximum(positives, 1e-6), 0.5, 5.0), dtype=torch.float32, device=device
    )
    print("positive target weights", dict(zip(TARGETS, positive_weight.cpu().tolist())))
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(model_config["backbone_lr"]),
            float(model_config["head_lr"]),
            float(model_config["weight_decay"]),
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(model_config["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema = ModelEMA(model)
    parallel_model = torch.nn.DataParallel(model) if torch.cuda.device_count() >= 2 else model

    best_score = -np.inf
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    bad_epochs = 0
    patience = int(model_config.get("early_stopping_patience", 3))
    minimum_epochs = int(model_config.get("minimum_epochs", 4))
    for epoch in range(int(model_config["epochs"])):
        parallel_model.train()
        running_loss = 0.0
        seen = 0
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = parallel_model(images, valid)
                loss = masked_soft_bce(logits, targets, weights, positive_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running_loss += float(loss.detach()) * len(images)
            seen += len(images)
            if (batch_index + 1) % 250 == 0:
                print(json.dumps({
                    "fold": fold,
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "batches": len(train_loader),
                    "elapsed_seconds": time.monotonic() - epoch_started,
                }), flush=True)
        scheduler.step()

        live_state = deepcopy(model.state_dict())
        model.load_state_dict(ema.state, strict=True)
        ids, predictions, truths, sample_weights, gold = _predict(parallel_model, valid_loader, device)
        model.load_state_dict(live_state, strict=True)
        addressed = sample_weights > 0
        binary_truths = (truths > 0.5).astype(np.uint8)
        pseudo_score, per_target = macro_auc(binary_truths, predictions, addressed)
        gold_score, gold_per_target = macro_auc(binary_truths, predictions, gold)
        selection_score = pseudo_score if not np.isfinite(gold_score) else 0.7 * pseudo_score + 0.3 * gold_score
        epoch_record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(seen, 1),
            "pseudo_auc": pseudo_score,
            "gold_auc": gold_score,
            "selection_auc": selection_score,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))
        if selection_score > best_score:
            best_score = selection_score
            bad_epochs = 0
            best_payload = {
                "model": deepcopy(ema.state),
                "model_config": persisted_model_config,
                "fold": fold,
                "epoch": epoch + 1,
                "selection_score": selection_score,
                "pseudo_auc": pseudo_score,
                "gold_auc": gold_score,
                "per_target_auc": dict(zip(TARGETS, per_target.tolist())),
                "gold_per_target_auc": dict(zip(TARGETS, gold_per_target.tolist())),
                "initialization": initialization,
                "ids": ids,
                "predictions": predictions,
                "truths": truths,
                "sample_weights": sample_weights,
                "gold": gold,
            }
        else:
            bad_epochs += 1
        if epoch + 1 >= minimum_epochs and bad_epochs >= patience:
            print(f"early stopping fold {fold} after epoch {epoch + 1}")
            break

    if best_payload is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    checkpoint_path = output_dir / f"fold_{fold}.pt"
    serializable = {key: value for key, value in best_payload.items() if key not in {"ids", "predictions", "truths", "sample_weights", "gold"}}
    torch.save(serializable, checkpoint_path)
    oof = pd.DataFrame({ID_COLUMN: best_payload["ids"], "fold": fold})
    for index, target in enumerate(TARGETS):
        oof[f"pred__{target}"] = best_payload["predictions"][:, index]
        oof[f"true__{target}"] = best_payload["truths"][:, index]
        oof[f"weight__{target}"] = best_payload["sample_weights"][:, index]
        oof[f"gold__{target}"] = best_payload["gold"][:, index].astype(np.uint8)
    oof_path = output_dir / f"oof_fold_{fold}.csv"
    oof.to_csv(oof_path, index=False)
    (output_dir / f"history_fold_{fold}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint_path), "oof": str(oof_path), "best_score": best_score}


def train_frozen_feature_folds(
    labels_path: str | Path,
    cache_paths: list[str | Path],
    output_dir: str | Path,
    model_config: dict[str, Any],
    seed: int = 20260905,
    gold_weight: float = 4.0,
    num_workers: int = 2,
    backbone_checkpoint: str | Path | None = None,
    backbone_checkpoint_license: str | None = None,
) -> list[dict[str, Any]]:
    """Extract a frozen CNN feature table once, then fit five lightweight MIL heads.

    Full CoAtNet backpropagation is too slow for the free Kaggle T4 budget. This
    path keeps the audited Raptor encoder intact, performs one center-triplet
    forward pass per study, and trains fold-specific attention heads from the
    resulting in-memory feature table. Checkpoints retain the ordinary
    ``StudyMILModel`` schema, so submission inference needs no special case.
    """
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("frozen feature training requires a Kaggle GPU runtime")
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(labels_path) if str(labels_path).endswith((".parquet", ".pq")) else pd.read_csv(labels_path)
    if "fold" not in labels:
        raise ValueError("labels file must contain a fold column")
    labels = labels.reset_index(drop=True)
    device = "cuda:0"
    persisted_model_config = {
        "backbone_name": model_config["backbone"],
        "input_size": int(model_config["image_size"]),
        "n_targets": len(TARGETS),
        "backbone_checkpoint": None,
        "last_blocks_trainable": 0,
        "pretrained": False,
    }
    initialization = {
        "requested_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
        "requested_checkpoint_license": backbone_checkpoint_license,
        "actual_source": "timm_pretrained" if backbone_checkpoint is None else "external_checkpoint",
        "actual_license": "Apache-2.0" if backbone_checkpoint is None else backbone_checkpoint_license,
        "training_mode": "frozen_encoder_single_pass_features",
    }
    try:
        model = StudyMILModel(
            **{
                **persisted_model_config,
                "backbone_checkpoint": backbone_checkpoint,
                "pretrained": backbone_checkpoint is None,
            }
        )
    except (RuntimeError, ValueError) as exc:
        if backbone_checkpoint is None:
            raise
        print(f"WARNING incompatible external checkpoint ({exc}); falling back to pinned timm initialization")
        initialization["actual_source"] = "timm_pretrained_fallback"
        initialization["actual_license"] = "Apache-2.0"
        initialization["fallback_reason"] = str(exc)
        model = StudyMILModel(**{**persisted_model_config, "pretrained": True})
    initialization["encoder_sha256"] = module_state_sha256(model.encoder)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model = model.to(device).eval()

    feature_data = CachedStudyDataset(labels, cache_paths, training=False, gold_weight=gold_weight)
    feature_loader = torch.utils.data.DataLoader(
        feature_data,
        batch_size=int(model_config.get("feature_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    parallel_model = torch.nn.DataParallel(model) if torch.cuda.device_count() >= 2 else model
    feature_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    feature_started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(feature_loader):
            images = batch["images"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoded, triplet_valid = parallel_model(images, valid, features_only=True)
            feature_rows.append(encoded.detach().cpu().numpy().astype(np.float16, copy=False))
            valid_rows.append(triplet_valid.detach().cpu().numpy().astype(bool, copy=False))
            if (batch_index + 1) % 50 == 0:
                print(json.dumps({
                    "stage": "feature_extraction",
                    "batch": batch_index + 1,
                    "batches": len(feature_loader),
                    "elapsed_seconds": time.monotonic() - feature_started,
                }), flush=True)
    features = np.concatenate(feature_rows)
    triplet_valid = np.concatenate(valid_rows)
    if features.shape[:2] != (len(labels), 6) or triplet_valid.shape != (len(labels), 6):
        raise RuntimeError(f"unexpected frozen feature shapes: {features.shape}, {triplet_valid.shape}")
    initialization["feature_extraction_seconds"] = time.monotonic() - feature_started
    initialization["feature_shape"] = list(features.shape)
    print(json.dumps({
        "stage": "feature_extraction_complete",
        "shape": list(features.shape),
        "elapsed_seconds": initialization["feature_extraction_seconds"],
    }), flush=True)

    target_values = labels.loc[:, list(TARGETS)].to_numpy(dtype=np.float32)
    confidence_values = np.column_stack([
        labels[f"{target}__confidence"].to_numpy(dtype=np.float32) for target in TARGETS
    ])
    gold_values = np.column_stack([
        labels[f"{target}__gold"].to_numpy(dtype=bool) for target in TARGETS
    ])
    training_weights = confidence_values * np.where(gold_values, gold_weight, 1.0)
    feature_tensor = torch.from_numpy(features)
    valid_tensor = torch.from_numpy(triplet_valid)
    target_tensor = torch.from_numpy(target_values)
    weight_tensor = torch.from_numpy(training_weights.astype(np.float32))
    gold_tensor = torch.from_numpy(gold_values)
    row_tensor = torch.arange(len(labels), dtype=torch.int64)
    full_dataset = torch.utils.data.TensorDataset(
        feature_tensor, valid_tensor, target_tensor, weight_tensor, gold_tensor, row_tensor
    )

    head_names = {name for name, _ in model.named_parameters() if not name.startswith("encoder.")}
    base_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in head_names
    }
    results: list[dict[str, Any]] = []
    for fold in range(5):
        seed_everything(seed + fold)
        model.reset_head()
        train_indices = np.flatnonzero(labels["fold"].to_numpy() != fold)
        valid_indices = np.flatnonzero(labels["fold"].to_numpy() == fold)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_dataset, train_indices.tolist()),
            batch_size=int(model_config.get("head_batch_size", 256)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_dataset, valid_indices.tolist()),
            batch_size=int(model_config.get("head_batch_size", 256)) * 2,
            shuffle=False,
            num_workers=0,
        )
        fold_targets = target_values[train_indices]
        fold_weights = training_weights[train_indices]
        positives = (fold_weights * fold_targets).sum(axis=0)
        negatives = (fold_weights * (1.0 - fold_targets)).sum(axis=0)
        positive_weight = torch.tensor(
            np.clip(negatives / np.maximum(positives, 1e-6), 0.5, 5.0), dtype=torch.float32, device=device
        )
        head_parameters = [parameter for name, parameter in model.named_parameters() if name in head_names]
        optimizer = torch.optim.AdamW(
            head_parameters,
            lr=float(model_config.get("frozen_head_lr", 1e-3)),
            weight_decay=float(model_config["weight_decay"]),
        )
        head_epochs = int(model_config.get("frozen_head_epochs", 40))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs)
        ema_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name in head_names}
        ema_decay = float(model_config.get("ema_decay", 0.995))
        best_score = -np.inf
        best_head: dict[str, Any] | None = None
        best_arrays: tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        history: list[dict[str, float]] = []
        bad_epochs = 0
        patience = int(model_config.get("frozen_head_patience", 6))
        minimum_epochs = int(model_config.get("frozen_head_minimum_epochs", 12))
        for epoch in range(head_epochs):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch_features, batch_valid, batch_targets, batch_weights, _, _ in train_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_valid = batch_valid.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_weights = batch_weights.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model.classify_features(batch_features, batch_valid)
                    loss = masked_soft_bce(logits, batch_targets, batch_weights, positive_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in ema_state:
                            ema_state[name].mul_(ema_decay).add_(parameter.detach(), alpha=1.0 - ema_decay)
                running_loss += float(loss.detach()) * len(batch_features)
                seen += len(batch_features)
            scheduler.step()

            live_head = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name in head_names}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in ema_state:
                        parameter.copy_(ema_state[name])
            model.eval()
            row_ids: list[int] = []
            predictions: list[np.ndarray] = []
            truths: list[np.ndarray] = []
            sample_weights: list[np.ndarray] = []
            gold_masks: list[np.ndarray] = []
            with torch.inference_mode():
                for batch_features, batch_valid, batch_targets, batch_weights, batch_gold, batch_rows in valid_loader:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model.classify_features(
                            batch_features.to(device, non_blocking=True),
                            batch_valid.to(device, non_blocking=True),
                        )
                    predictions.append(torch.sigmoid(logits).float().cpu().numpy())
                    truths.append(batch_targets.numpy())
                    sample_weights.append(batch_weights.numpy())
                    gold_masks.append(batch_gold.numpy())
                    row_ids.extend(batch_rows.tolist())
            for name, parameter in model.named_parameters():
                if name in live_head:
                    parameter.data.copy_(live_head[name])
            prediction_values = np.concatenate(predictions)
            truth_values = np.concatenate(truths)
            sample_weight_values = np.concatenate(sample_weights)
            gold_mask_values = np.concatenate(gold_masks)
            addressed = sample_weight_values > 0
            binary_truths = (truth_values > 0.5).astype(np.uint8)
            pseudo_score, per_target = macro_auc(binary_truths, prediction_values, addressed)
            gold_score, gold_per_target = macro_auc(binary_truths, prediction_values, gold_mask_values)
            selection_score = pseudo_score if not np.isfinite(gold_score) else 0.7 * pseudo_score + 0.3 * gold_score
            record = {
                "epoch": epoch + 1,
                "loss": running_loss / max(seen, 1),
                "pseudo_auc": pseudo_score,
                "gold_auc": gold_score,
                "selection_auc": selection_score,
            }
            history.append(record)
            print(json.dumps({"fold": fold, **record}), flush=True)
            if selection_score > best_score:
                best_score = selection_score
                bad_epochs = 0
                best_head = {name: value.detach().cpu().clone() for name, value in ema_state.items()}
                best_arrays = (row_ids, prediction_values, truth_values, sample_weight_values, gold_mask_values)
                best_metrics = (pseudo_score, gold_score, per_target, gold_per_target, epoch + 1)
            else:
                bad_epochs += 1
            if epoch + 1 >= minimum_epochs and bad_epochs >= patience:
                print(f"early stopping frozen head fold {fold} after epoch {epoch + 1}")
                break

        if best_head is None or best_arrays is None:
            raise RuntimeError(f"frozen head fold {fold} did not produce a valid checkpoint")
        pseudo_score, gold_score, per_target, gold_per_target, best_epoch = best_metrics
        checkpoint_path = output_dir / f"fold_{fold}.pt"
        torch.save({
            "model": {**base_state, **best_head},
            "model_config": persisted_model_config,
            "fold": fold,
            "epoch": best_epoch,
            "selection_score": best_score,
            "pseudo_auc": pseudo_score,
            "gold_auc": gold_score,
            "per_target_auc": dict(zip(TARGETS, per_target.tolist())),
            "gold_per_target_auc": dict(zip(TARGETS, gold_per_target.tolist())),
            "initialization": initialization,
        }, checkpoint_path)
        row_ids, prediction_values, truth_values, sample_weight_values, gold_mask_values = best_arrays
        oof = pd.DataFrame({ID_COLUMN: labels.iloc[row_ids][ID_COLUMN].astype(str).tolist(), "fold": fold})
        for index, target in enumerate(TARGETS):
            oof[f"pred__{target}"] = prediction_values[:, index]
            oof[f"true__{target}"] = truth_values[:, index]
            oof[f"weight__{target}"] = sample_weight_values[:, index]
            oof[f"gold__{target}"] = gold_mask_values[:, index].astype(np.uint8)
        oof_path = output_dir / f"oof_fold_{fold}.csv"
        oof.to_csv(oof_path, index=False)
        (output_dir / f"history_fold_{fold}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        results.append({"checkpoint": str(checkpoint_path), "oof": str(oof_path), "best_score": best_score})
    return results
