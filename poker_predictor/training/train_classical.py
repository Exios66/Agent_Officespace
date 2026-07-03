"""Train the classical multi-head baseline on PokerBench preflop data."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loaders import load_pokerbench_preflop
from ..data.schemas import PreflopSample
from ..features.build import build_feature_matrix, canonical_action_label
from ..models.baselines import MultiHeadModel, train_action_head, train_villain_fold_head
from .labels import villain_fold_label

log = logging.getLogger(__name__)


def _try_trackio(project: str):
    try:
        import trackio  # type: ignore

        return trackio.init(project=project)
    except Exception:  # pragma: no cover - optional
        return None


def prepare_training_frame(
    samples: list[PreflopSample],
) -> tuple[pd.DataFrame, list[str | None], list[int]]:
    X, raw_y = build_feature_matrix(samples)
    y = [canonical_action_label(v) for v in raw_y]
    villain_y = [villain_fold_label(s) for s in samples]
    return X, y, villain_y


def train(
    output_dir: str | Path = "artifacts/classical",
    model_kind: str = "lightgbm",
    limit: int | None = None,
    val_frac: float = 0.1,
    seed: int = 7,
    samples: list[PreflopSample] | None = None,
) -> MultiHeadModel:
    """Full training entrypoint.

    If ``samples`` is None, PokerBench preflop train split is downloaded.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if samples is None:
        log.info("Loading PokerBench preflop train split (limit=%s) ...", limit)
        samples = load_pokerbench_preflop(split="train", limit=limit)
    log.info("Loaded %d samples in %.2fs", len(samples), time.time() - t0)

    X, y, villain_y = prepare_training_frame(samples)
    mask = [v is not None for v in y]
    X = X.loc[mask].reset_index(drop=True)
    y_clean = [v for v in y if v is not None]
    villain_y = [v for v, m in zip(villain_y, mask, strict=False) if m]

    X_train, X_val, y_train, y_val, vy_train, vy_val = train_test_split(
        X, y_clean, villain_y, test_size=val_frac, random_state=seed, stratify=y_clean
    )

    run = _try_trackio(project="poker-preflop")

    log.info("Training action head (%s) on %d rows ...", model_kind, len(X_train))
    action_model, encoder = train_action_head(X_train, y_train, kind=model_kind)

    log.info("Training villain-fold head ...")
    villain_model = train_villain_fold_head(X_train, vy_train, kind=model_kind)

    model = MultiHeadModel(
        action_model=action_model,
        action_encoder=encoder,
        villain_fold_model=villain_model,
        feature_names=list(X.columns),
        meta={"model_kind": model_kind, "n_train": len(X_train), "n_val": len(X_val)},
    )

    val_acc = float(np.mean(action_model.predict(X_val) == encoder.transform(y_val)))
    log.info("Validation top-1 accuracy: %.4f", val_acc)

    if run is not None:
        try:
            import trackio  # type: ignore

            trackio.log({"val_top1_acc": val_acc, "n_train": len(X_train), "n_val": len(X_val)})
            trackio.finish()
        except Exception:  # pragma: no cover
            pass

    save_path = output / "multihead.joblib"
    model.save(save_path)
    log.info("Saved model to %s", save_path)
    return model
