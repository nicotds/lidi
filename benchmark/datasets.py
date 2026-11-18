"""Datasets del benchmark: generadores sintéticos (1M filas, semilla 42) y lectura de los reales.

Cada sintético es una sola columna con 2 % de NaN. Con la misma semilla y el mismo
tamaño se obtiene siempre exactamente el mismo arreglo.

Uso:  python -m benchmark.datasets      (escribe tablas/perfil_sinteticos.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SINTETICOS = ("normal", "lognormal", "uniforme", "bimodal", "normal_outliers", "ceros")


def generar(nombre: str, n: int = 1_000_000, semilla: int = 42, pct_nan: float = 0.02) -> np.ndarray:
    """Devuelve un ndarray float64 de n valores de la distribución `nombre`, con pct_nan de NaN."""
    if nombre not in SINTETICOS:
        raise ValueError(f"distribución {nombre!r} no válida; opciones: {SINTETICOS}")
    rng = np.random.default_rng(semilla)

    if nombre == "normal":
        x = rng.normal(50.0, 10.0, n)
    elif nombre == "lognormal":                      # sesgada, tipo ingresos
        x = rng.lognormal(mean=10.0, sigma=0.6, size=n)
    elif nombre == "uniforme":
        x = rng.uniform(0.0, 100.0, n)
    elif nombre == "bimodal":                        # mezcla 50/50 de dos normales
        x = np.where(rng.random(n) < 0.5, rng.normal(30.0, 5.0, n), rng.normal(70.0, 5.0, n))
    elif nombre == "normal_outliers":                # 0,1 % de valores a 50 desviaciones
        x = rng.normal(50.0, 10.0, n)
        k = int(round(0.001 * n))
        idx = rng.choice(n, k, replace=False)
        x[idx] = 50.0 + rng.choice([-1.0, 1.0], k) * 50.0 * 10.0
    else:                                            # ceros: 70 % de ceros, el resto normal positivo
        x = np.abs(rng.normal(50.0, 10.0, n))
        x[rng.choice(n, int(round(0.7 * n)), replace=False)] = 0.0

    x = np.asarray(x, dtype="float64")
    x[rng.choice(n, int(round(pct_nan * n)), replace=False)] = np.nan
    return x


def perfil(x: np.ndarray, nombre: str) -> dict:
    s = pd.Series(x)
    return {
        "dataset": nombre, "n": int(s.size), "pct_nan": float(s.isna().mean() * 100),
        "pct_ceros": float((s == 0).mean() * 100),
        "media": float(s.mean()), "desv": float(s.std()),
        "minimo": float(s.min()), "maximo": float(s.max()), "asimetria": float(s.skew()),
    }


def perfil_sinteticos(n: int = 1_000_000, semilla: int = 42, pct_nan: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame([perfil(generar(d, n, semilla, pct_nan), d) for d in SINTETICOS])


def cargar_real(ruta, columnas) -> dict:
    """Lee solo las columnas pedidas de un Parquet. Devuelve {columna: ndarray float64}."""
    df = pd.read_parquet(ruta, columns=list(columnas))
    return {c: df[c].to_numpy(dtype="float64") for c in columnas}


if __name__ == "__main__":
    destino = Path("resultados/actividad4/tablas/perfil_sinteticos.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla = perfil_sinteticos()
    tabla.to_csv(destino, index=False)
    pd.set_option("display.width", 200)
    print(tabla.round(3).to_string(index=False))
    print("\nguardado en", destino)
