"""Estima el tiempo de la corrida completa a partir de un piloto.

  python scripts/estimar_benchmark.py --piloto resultados/actividad4/_piloto/resultados_benchmark.parquet --n-piloto 100000

Supone que el tiempo por caso crece linealmente con el número de filas (aproximación: los
cuantiles y kmeans pueden crecer algo más) y lo interpola/extrapola linealmente en los bins.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from benchmark.runner import cargar_config, lista_metodos  # noqa: E402


def filas_dataset(nombre: str, ruta: str) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(RAIZ / ruta).metadata.num_rows
    except Exception:
        return {"1M": 1_000_000, "5M": 5_000_000}[nombre]


def tiempo_por_caso(piloto: pd.DataFrame) -> dict:
    """{metodo: (bins_array, tiempos_array)} con la media de t_caso_s por bins."""
    casos = piloto.drop_duplicates(["dataset", "columna", "metodo", "bins_solicitados"])
    out = {}
    for m, g in casos.groupby("metodo"):
        h = g.groupby(g["bins_solicitados"].fillna(0))["t_caso_s"].mean().sort_index()
        out[m] = (h.index.to_numpy(float), h.to_numpy(float))
    return out


def tiempo_en(bins: float, puntos) -> float:
    xs, ys = puntos
    if len(xs) == 1:
        return float(ys[0])
    t = float(np.interp(bins, xs, ys))
    if bins > xs[-1]:
        t = float(ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (bins - xs[-1]))
    elif bins < xs[0]:
        t = float(ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (bins - xs[0]))
    return max(t, float(ys.min()) * 0.5)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--piloto", required=True)
    a.add_argument("--n-piloto", type=int, required=True, help="filas usadas en el piloto")
    a.add_argument("--config", default=str(RAIZ / "benchmark" / "config.yaml"))
    args = a.parse_args(argv)

    cfg = cargar_config(args.config)
    pt = tiempo_por_caso(pd.read_parquet(args.piloto))
    metodos = lista_metodos(cfg)
    faltan = [m for m in metodos + cfg["metodos"]["controles"] if m not in pt]
    if faltan:
        print("AVISO: el piloto no incluye estos métodos, no se estiman:", faltan)

    objetivos = [(d, cfg["sinteticos"]["n_filas"], 1, "sintético") for d in cfg["sinteticos"]["distribuciones"]]
    objetivos += [(d, filas_dataset(d, r), len(cfg["reales"]["columnas"]), "real")
                  for d, r in cfg["reales"]["datasets"].items()]

    filas = []
    for ds, n, ncol, tipo in objetivos:
        factor = n / args.n_piloto
        for m in metodos:
            if m in pt:
                s = sum(tiempo_en(b, pt[m]) for b in cfg["bins"]) * ncol * factor
                filas.append({"dataset": ds, "tipo": tipo, "metodo": m, "segundos": s})
        for m in cfg["metodos"]["controles"]:
            if m in pt:
                filas.append({"dataset": ds, "tipo": tipo, "metodo": m,
                              "segundos": tiempo_en(0, pt[m]) * ncol * factor})
    est = pd.DataFrame(filas)

    print("\nEstimación por dataset (minutos):")
    por_ds = est.groupby(["tipo", "dataset"])["segundos"].sum().div(60).round(1)
    print(por_ds.to_string())
    print("\nEstimación por método sobre el 5M real (minutos):")
    if (est["dataset"] == "5M").any():
        print(est[est["dataset"] == "5M"].set_index("metodo")["segundos"].div(60).round(1).to_string())
    print(f"\nTOTAL estimado: {est['segundos'].sum() / 3600:.2f} h")
    print("Es una estimación lineal: kmeans y cuantiles pueden crecer más que lineal con las filas.")


if __name__ == "__main__":
    main()
