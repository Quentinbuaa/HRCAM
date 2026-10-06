"""Numerically explicit RCAM components and bounded comparison candidates."""
import numpy as np


def heatmap_metrics(first, second, mask, eps=1e-8):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or mask.shape != first.shape:
        raise ValueError("Heatmaps and mask must share a two-dimensional shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all() or min(first.min(), second.min()) < 0:
        raise ValueError("Heatmaps must be finite and nonnegative")
    a, b = first.ravel(), second.ravel()
    valid = bool(a.sum() > eps and b.sum() > eps and a.std() > eps and b.std() > eps)
    if not valid:
        return {**dict.fromkeys(["d1", "d2", "d3", "d4", "jsd", "foreground_shift", "background_increase"], np.nan), "valid_cam": False}
    yy, xx = np.indices(first.shape)
    p0, q0 = a / a.sum(), b / b.sum()
    p, q = (a + eps) / (a + eps).sum(), (b + eps) / (b + eps).sum()
    c1 = np.array([p0 @ yy.ravel(), p0 @ xx.ravel()])
    c2 = np.array([q0 @ yy.ravel(), q0 @ xx.ravel()])
    d1 = np.linalg.norm(c1 - c2) / np.hypot(*first.shape)
    d2 = max(0., float(np.sum(p * np.log(p / q))))
    count = int(np.ceil(0.2 * a.size))
    sa, sb = np.zeros(a.size, bool), np.zeros(b.size, bool)
    sa[np.argsort(-a, kind="stable")[:count]] = True
    sb[np.argsort(-b, kind="stable")[:count]] = True
    d3 = 1. - (sa & sb).sum() / (sa | sb).sum()
    d4 = float(np.clip(1. - np.corrcoef(a, b)[0, 1], 0., 2.))
    m = (p + q) / 2
    jsd = float(np.clip((np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / (2 * np.log(2)), 0., 1.))
    fg1, fg2 = p0 @ mask.ravel(), q0 @ mask.ravel()
    return {"d1": float(d1), "d2": d2, "d3": float(d3), "d4": d4,
            "jsd": jsd, "foreground_shift": float(abs(fg1 - fg2)),
            "background_increase": float(max(0., fg1 - fg2)), "valid_cam": True}


def fit_minmax(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Expected finite N x 4 calibration components")
    return {"minimum": values.min(axis=0).tolist(), "maximum": values.max(axis=0).tolist()}


def apply_minmax(values, parameters):
    lo, hi = np.asarray(parameters["minimum"]), np.asarray(parameters["maximum"])
    span = hi - lo
    # A constant calibration component provides no calibration discrimination.
    scaled = np.divide(np.asarray(values) - lo, span, out=np.zeros_like(np.asarray(values), dtype=float), where=span > 0)
    clipped = (scaled < 0) | (scaled > 1)
    return np.clip(scaled, 0., 1.), clipped
