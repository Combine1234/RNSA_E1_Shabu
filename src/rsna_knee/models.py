from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required on the Kaggle training/inference runtime") from exc
    return torch


def _unwrap_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "ema"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    return payload


def _strip_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()}


def module_state_sha256(module) -> str:
    """Hash tensor names, shapes, dtypes and bytes without serialisation metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _compatible_encoder_state(module, raw_state: dict[str, Any]) -> dict[str, Any]:
    target = module.state_dict()
    matched: dict[str, Any] = {}
    prefixes = ("module.", "model.", "encoder.", "backbone.")
    for raw_name, value in raw_state.items():
        candidates = {raw_name}
        changed = True
        while changed:
            changed = False
            for name in list(candidates):
                for prefix in prefixes:
                    if name.startswith(prefix) and name[len(prefix) :] not in candidates:
                        candidates.add(name[len(prefix) :])
                        changed = True
                for marker in (".encoder.", ".backbone."):
                    if marker in name:
                        candidates.add(name.split(marker, 1)[1])
        for name in candidates:
            if name in target and getattr(value, "shape", None) == target[name].shape:
                matched[name] = value
                break
    return matched


class StudyMILModel:
    """Shared 2.5D encoder with target-specific attention over six acquisition slots."""

    def __new__(cls, *args, **kwargs):
        torch = _torch()

        class _Impl(torch.nn.Module):
            def __init__(
                self,
                backbone_name: str,
                input_size: int,
                n_targets: int = 12,
                backbone_checkpoint: str | Path | None = None,
                last_blocks_trainable: int = -1,
                pretrained: bool = False,
            ):
                super().__init__()
                import timm

                self.input_size = int(input_size)
                self.n_targets = int(n_targets)
                self.encoder = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    # Official DINOv2 weights keep their final layer as
                    # ``norm`` and require token pooling. Requesting average
                    # pooling makes timm rewrite it to ``fc_norm`` before the
                    # pretrained state is loaded. CNN/CoAtNet backbones keep
                    # their standard average pooling.
                    global_pool="token" if "dinov2" in backbone_name.casefold() else "avg",
                    img_size=self.input_size,
                )
                feature_dim = int(getattr(self.encoder, "num_features"))
                cfg = getattr(self.encoder, "pretrained_cfg", {}) or {}
                mean = cfg.get("mean", (0.485, 0.456, 0.406))
                std = cfg.get("std", (0.229, 0.224, 0.225))
                self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=True)
                self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=True)
                self.slot_embedding = torch.nn.Embedding(6, feature_dim)
                self.key = torch.nn.Linear(feature_dim, feature_dim)
                self.query = torch.nn.Parameter(torch.randn(n_targets, feature_dim) * (feature_dim ** -0.5))
                self.head_weight = torch.nn.Parameter(torch.randn(n_targets, feature_dim) * (feature_dim ** -0.5))
                self.head_bias = torch.nn.Parameter(torch.zeros(n_targets))
                self.dropout = torch.nn.Dropout(0.15)
                if backbone_checkpoint:
                    self.load_backbone(backbone_checkpoint)
                self.configure_trainable(last_blocks_trainable)

            def reset_head(self) -> None:
                """Reinitialize only the study-level attention/classification head."""
                self.slot_embedding.reset_parameters()
                self.key.reset_parameters()
                torch.nn.init.normal_(self.query, std=self.query.shape[-1] ** -0.5)
                torch.nn.init.normal_(self.head_weight, std=self.head_weight.shape[-1] ** -0.5)
                torch.nn.init.zeros_(self.head_bias)

            def load_backbone(self, path: str | Path) -> None:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = _compatible_encoder_state(self.encoder, _unwrap_state(payload))
                minimum = max(10, int(0.5 * len(self.encoder.state_dict())))
                if len(state) < minimum:
                    raise ValueError(
                        f"checkpoint {path} matched only {len(state)} encoder tensors; at least {minimum} required"
                    )
                missing, _ = self.encoder.load_state_dict(state, strict=False)
                print(f"loaded encoder checkpoint {path}: {len(state)} tensors, {len(missing)} missing")

            def configure_trainable(self, last_blocks: int) -> None:
                if last_blocks < 0:
                    for parameter in self.encoder.parameters():
                        parameter.requires_grad = True
                    return
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False
                blocks = getattr(self.encoder, "blocks", None)
                if blocks is not None and last_blocks > 0:
                    for block in list(blocks)[-last_blocks:]:
                        for parameter in block.parameters():
                            parameter.requires_grad = True
                    for name in ("norm", "fc_norm"):
                        module = getattr(self.encoder, name, None)
                        if module is not None:
                            for parameter in module.parameters():
                                parameter.requires_grad = True

            def parameter_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
                encoder = [p for p in self.encoder.parameters() if p.requires_grad]
                encoder_ids = {id(p) for p in encoder}
                head = [p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids]
                return [
                    {"params": encoder, "lr": backbone_lr, "weight_decay": weight_decay},
                    {"params": head, "lr": head_lr, "weight_decay": weight_decay},
                ]

            def encode_slots(self, images):
                # One adjacent 3-slice RGB view for each of six acquisition slots.
                batch, slots, slices, height, width = images.shape
                if slots != 6 or slices != 3:
                    raise ValueError(f"expected [B,6,3,H,W], received {tuple(images.shape)}")
                triplets = images.reshape(batch * slots, 3, height, width)
                if (height, width) != (self.input_size, self.input_size):
                    triplets = torch.nn.functional.interpolate(
                        triplets, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
                    )
                triplets = (triplets - self.image_mean) / self.image_std
                features = self.encoder(triplets)
                if features.ndim > 2:
                    features = features.flatten(2).mean(-1)
                return features.reshape(batch, slots, -1)

            def classify_features(self, features, triplet_valid):
                slots = features.shape[1]
                slot_ids = torch.arange(slots, device=features.device)
                features = features + self.slot_embedding(slot_ids)[None]
                keys = self.key(features)
                attention = torch.einsum("bnd,td->btn", keys, self.query) / (features.shape[-1] ** 0.5)
                attention = attention.masked_fill(~triplet_valid[:, None, :], -1e4)
                all_missing = ~triplet_valid.any(dim=1)
                attention = torch.softmax(attention, dim=-1)
                pooled = torch.einsum("btn,bnd->btd", attention, features)
                pooled = self.dropout(pooled)
                logits = (pooled * self.head_weight[None]).sum(dim=-1) + self.head_bias[None]
                return torch.where(all_missing[:, None], torch.zeros_like(logits), logits)

            def forward(self, images, valid, features_only: bool = False):
                features = self.encode_slots(images)
                triplet_valid = valid.all(dim=-1)
                if features_only:
                    return features, triplet_valid
                return self.classify_features(features, triplet_valid)

        return _Impl(*args, **kwargs)


def load_study_model(checkpoint: str | Path, device: str = "cpu"):
    torch = _torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint {checkpoint} is missing model_config")
    model = StudyMILModel(**config)
    state = _strip_prefix(_unwrap_state(payload), "module.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload
