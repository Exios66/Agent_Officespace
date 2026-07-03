"""PyTorch training loop for the tabular MLP baseline (optional).

Kept intentionally minimal: this is a second baseline, not the primary model.
Trackio logging is best-effort.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..features.build import canonical_action_label
from ..models.torch_mlp import TorchMLPConfig, build_model, has_torch

log = logging.getLogger(__name__)


def _tensorize(X: pd.DataFrame, y_enc: np.ndarray, villain_y: np.ndarray):
    import torch

    hand_idx = torch.from_numpy(X["hand_class_idx"].to_numpy(dtype=np.int64))
    pos_idx = torch.from_numpy(X["pos_idx"].to_numpy(dtype=np.int64))
    cont_cols = [c for c in X.columns if c not in ("hand_class_idx", "pos_idx")]
    x_cont = torch.from_numpy(X[cont_cols].to_numpy(dtype=np.float32))
    y_t = torch.from_numpy(y_enc.astype(np.int64))
    vy_t = torch.from_numpy(villain_y.astype(np.float32))
    return x_cont, hand_idx, pos_idx, y_t, vy_t, cont_cols


def train_torch(
    X: pd.DataFrame,
    y_raw: list[str | None],
    villain_y: list[int],
    output_dir: str | Path = "artifacts/torch",
    cfg: TorchMLPConfig | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    if not has_torch():
        raise RuntimeError("torch not installed; install extras: pip install '.[torch]'")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = cfg or TorchMLPConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mask = [v is not None for v in y_raw]
    X = X.loc[mask].reset_index(drop=True)
    y_clean = [canonical_action_label(v) for v in y_raw if v is not None]
    villain_y = [v for v, m in zip(villain_y, mask, strict=False) if m]

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y_clean)
    vy = np.asarray(villain_y, dtype=np.int64)

    X_tr, X_val, y_tr, y_val, vy_tr, vy_val = train_test_split(
        X, y_enc, vy, test_size=0.1, random_state=seed, stratify=y_enc
    )

    x_cont_tr, h_tr, p_tr, y_tr_t, vy_tr_t, cont_cols = _tensorize(X_tr, y_tr, vy_tr)
    x_cont_val, h_val, p_val, y_val_t, vy_val_t, _ = _tensorize(X_val, y_val, vy_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        n_continuous=x_cont_tr.shape[1], n_actions=len(enc.classes_), cfg=cfg
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    ds = TensorDataset(x_cont_tr, h_tr, p_tr, y_tr_t, vy_tr_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, hb, pb, yb, vb in dl:
            xb, hb, pb, yb, vb = (t.to(device) for t in (xb, hb, pb, yb, vb))
            out = model(xb, hb, pb)
            loss_action = ce(out["action_logits"], yb)
            vf_mask = (vb >= 0).float()
            loss_vf = (bce(out["villain_fold_logit"], vb.clamp(min=0.0)) * vf_mask).sum() / vf_mask.sum().clamp(min=1.0)
            loss = loss_action + 0.5 * loss_vf
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]

        model.eval()
        with torch.no_grad():
            out_val = model(x_cont_val.to(device), h_val.to(device), p_val.to(device))
            preds = out_val["action_logits"].argmax(dim=-1).cpu().numpy()
            val_acc = float((preds == y_val).mean())
        log.info("epoch %d  train_loss=%.4f  val_acc=%.4f", epoch, total / max(n, 1), val_acc)

    save_path = output / "torch_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder_classes": list(enc.classes_),
            "cont_cols": cont_cols,
        },
        save_path,
    )
    return {"val_acc": val_acc, "path": str(save_path)}
