"""Task-specific performance metrics from joined ground truth only.

Never fabricates a score.  If labels are insufficient the snapshot status
is ``awaiting_labels`` and metric fields stay ``None``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)
    return (lo, hi)


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    multiclass: bool = False,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"n_labeled": 0}

    acc = float((y_true == y_pred).mean())
    lo, hi = wilson_interval(int((y_true == y_pred).sum()), n)

    if not multiclass:
        # Map to {0,1} using unique labels if needed
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(labels) == 2:
            pos = labels.max()
            yt = (y_true == pos).astype(int)
            yp = (y_pred == pos).astype(int)
            prec, rec, f1 = _binary_prf(yt, yp)
        else:
            prec, rec, f1 = _binary_prf(
                (y_true.round() == 1).astype(int),
                (y_pred.round() == 1).astype(int),
            )
    else:
        # Macro precision/recall/F1
        classes = np.unique(y_true)
        precs, recs, f1s = [], [], []
        for cls in classes:
            yt = (y_true == cls).astype(int)
            yp = (y_pred == cls).astype(int)
            p, r, f = _binary_prf(yt, yp)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
        prec = float(np.mean(precs)) if precs else 0.0
        rec = float(np.mean(recs)) if recs else 0.0
        f1 = float(np.mean(f1s)) if f1s else 0.0

    result = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "n_labeled": n,
        "accuracy_ci": [round(lo, 4), round(hi, 4)],
    }
    if scores is not None and len(scores) == n and not multiclass:
        try:
            result["auc"] = round(_auc(y_true, scores), 6)
        except Exception:
            result["auc"] = None
    return result


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney AUC without sklearn."""
    labels = np.unique(y_true)
    if len(labels) != 2:
        raise ValueError("AUC requires binary labels")
    pos_label = labels.max()
    pos = scores[y_true == pos_label]
    neg = scores[y_true != pos_label]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires both classes")
    # Rank-based
    n_pos, n_neg = len(pos), len(neg)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    pos_ranks = ranks[:n_pos]
    u = pos_ranks.sum() - n_pos * (n_pos - 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n_labeled": 0}
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "n_labeled": n,
    }


def performance_ratio(
    current: Optional[float],
    baseline: Optional[float],
    higher_is_better: bool = True,
) -> Optional[float]:
    """current / baseline, inverted when lower is better (MAE/RMSE)."""
    if current is None or baseline is None or baseline == 0:
        return None
    if higher_is_better:
        return current / baseline
    # lower-is-better: baseline/current so a rise in MAE yields ratio < 1
    if current == 0:
        return 1.0
    return baseline / current
