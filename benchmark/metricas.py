"""Métricas del benchmark.

Todas se calculan solo sobre pares finitos (se descartan los NaN). Las normalizadas dan
NaN, sin dividir por cero, cuando el rango o la desviación del original es 0.

- NMAE = MAE / rango;  NRMSE = RMSE / desviación;  err_max_norm = máximo / rango.
- La desviación es la poblacional (ddof=0) en todas las métricas.
- W1 normalizado = wasserstein_distance / desviación del original.
- dist_hist = L1 entre histogramas normalizados de 50 bins sobre el rango del original.
- Distribución: si se pasa `idx`, se calcula sobre esa muestra fija (se usa en el 5M).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

N_BINS_HIST = 50


def _pares(original, reconstruido):
    o = np.asarray(original, dtype="float64")
    r = np.asarray(reconstruido, dtype="float64")
    if o.shape != r.shape:
        raise ValueError(f"longitudes distintas: {o.shape} vs {r.shape}")
    m = np.isfinite(o) & np.isfinite(r)
    return o[m], r[m]


def _div(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) and b > 0 else float("nan")


def error_puntual(original, reconstruido) -> dict:
    """mae, rmse, err_max (crudos) y nmae, nrmse, err_max_norm (normalizados)."""
    o, r = _pares(original, reconstruido)
    if o.size == 0:
        return {k: float("nan") for k in ("mae", "rmse", "err_max", "nmae", "nrmse", "err_max_norm")}
    d = np.abs(r - o)
    mae, rmse, emax = float(d.mean()), float(np.sqrt((d ** 2).mean())), float(d.max())
    rango, sd = float(o.max() - o.min()), float(o.std())
    return {"mae": mae, "rmse": rmse, "err_max": emax,
            "nmae": _div(mae, rango), "nrmse": _div(rmse, sd), "err_max_norm": _div(emax, rango)}


def dist_histograma(o: np.ndarray, r: np.ndarray, n_bins: int = N_BINS_HIST) -> float:
    lo, hi = float(o.min()), float(o.max())
    if not hi > lo:
        return float("nan")
    bordes = np.linspace(lo, hi, n_bins + 1)
    ho, _ = np.histogram(o, bins=bordes)
    hr, _ = np.histogram(r, bins=bordes)
    if ho.sum() == 0 or hr.sum() == 0:
        return float("nan")
    return float(np.abs(ho / ho.sum() - hr / hr.sum()).sum())


def metricas_distribucion(original, reconstruido, idx=None) -> dict:
    """w1_norm, ks y dist_hist. `idx` (opcional): índices de una muestra fija."""
    o = np.asarray(original, dtype="float64")
    r = np.asarray(reconstruido, dtype="float64")
    if idx is not None:
        o, r = o[idx], r[idx]
    o, r = _pares(o, r)
    if o.size == 0:
        return {"w1_norm": float("nan"), "ks": float("nan"), "dist_hist": float("nan")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # ks_2samp avisa al pasar de cálculo exacto a asintótico
        ks = float(ks_2samp(o, r).statistic)
    return {"w1_norm": _div(wasserstein_distance(o, r), float(o.std())),
            "ks": ks, "dist_hist": dist_histograma(o, r)}


def error_correlacion(original: pd.DataFrame, reconstruido: pd.DataFrame) -> float:
    """Máximo de |corr(reconstruido) - corr(original)| fuera de la diagonal (Pearson)."""
    if original.shape[1] < 2:
        return float("nan")
    dif = (reconstruido.corr() - original.corr()).to_numpy()
    fuera = ~np.eye(dif.shape[0], dtype=bool)
    return float(np.nanmax(np.abs(dif[fuera]))) if np.isfinite(dif[fuera]).any() else float("nan")


def bits_por_valor(bins_reales: int) -> int:
    return math.ceil(math.log2(bins_reales + 1))


def razon_compresion(bits: float) -> float:
    return 64.0 / bits


def pct_fuera_rango(x, limites) -> float:
    """% de valores no NaN fuera de [limites[0], limites[-1]]. Sin límites (controles): 0."""
    if limites is None:
        return 0.0
    x = np.asarray(x, dtype="float64")
    v = x[~np.isnan(x)]
    if v.size == 0:
        return float("nan")
    return float(np.mean((v < limites[0]) | (v > limites[-1])) * 100)
