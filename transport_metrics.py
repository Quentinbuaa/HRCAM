"""Spatial and mask-aware Wasserstein candidates on a fixed mass-preserving grid."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

LOCAL_DEPS = Path(__file__).resolve().parent / "deps_py314"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import ot


def probability(heatmap):
    a = np.asarray(heatmap, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or a.min() < 0 or a.sum() <= 0:
        raise ValueError("Expected a finite nonnegative nonzero 2D heatmap")
    return a / a.sum()


def pool_mass(a, grid):
    h, w = a.shape
    if grid < 2 or h % grid or w % grid:
        raise ValueError("Grid must divide both heatmap dimensions and be >= 2")
    return a.reshape(grid, h//grid, grid, w//grid).sum(axis=(1, 3)).ravel()


@lru_cache(maxsize=4)
def cost_matrices(grid):
    yy, xx = np.meshgrid((np.arange(grid)+.5)/grid, (np.arange(grid)+.5)/grid, indexing="ij")
    points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    spatial = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1) / np.sqrt(2)
    regions = np.repeat([0., 1.], grid*grid)
    semantic = (np.tile(spatial, (2, 2)) + abs(regions[:, None]-regions[None, :])) / 2
    return np.ascontiguousarray(spatial), np.ascontiguousarray(semantic)


def exact_transport(p, q, cost, return_plan=False):
    support = (p > 0) | (q > 0)
    a, b = np.ascontiguousarray(p[support]), np.ascontiguousarray(q[support])
    c = np.ascontiguousarray(cost[np.ix_(support, support)])
    if return_plan:
        plan, log = ot.emd(a, b, c, numThreads=1, log=True)
        if log.get("warning"):
            raise RuntimeError(log["warning"])
        np.testing.assert_allclose(plan.sum(1), a, atol=1e-9)
        np.testing.assert_allclose(plan.sum(0), b, atol=1e-9)
        return float(np.sum(plan*c)), plan
    value, log = ot.emd2(a, b, c, numThreads=1, log=True)
    if log.get("warning"):
        raise RuntimeError(log["warning"])
    value = float(value)
    if not -1e-9 <= value <= 1+1e-9:
        raise ArithmeticError("Transport outside expected bounds")
    return value


def distances(first, second, mask, grid=14):
    p, q = probability(first), probability(second)
    mask = np.asarray(mask)
    if p.shape != q.shape or mask.shape != p.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Heatmaps must align with a binary foreground mask")
    fg = mask.astype(bool)
    a, b = pool_mass(p, grid), pool_mass(q, grid)
    # Split semantic mass before pooling to preserve thin foreground boundaries.
    pa = np.concatenate([pool_mass(p * ~fg, grid), pool_mass(p * fg, grid)])
    qa = np.concatenate([pool_mass(q * ~fg, grid), pool_mass(q * fg, grid)])
    spatial_cost, semantic_cost = cost_matrices(grid)
    return {"spatial_w1": exact_transport(a, b, spatial_cost),
            "BAT": exact_transport(pa, qa, semantic_cost),
            "foreground_mass_change": float(q[fg].sum()-p[fg].sum())}
