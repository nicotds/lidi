"""Resumen rápido de los sintéticos: NRMSE de prueba por método y bins, ranking medio y generalización.

  python scripts/resumen_sinteticos.py
Guarda resultados/actividad4/tablas/resumen_sinteticos_nrmse.csv y ranking_sinteticos.csv
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "resultados" / "actividad4" / "crudo" / "resultados_benchmark.parquet"
TABLAS = RAIZ / "resultados" / "actividad4" / "tablas"
SINTETICOS = ["normal", "lognormal", "uniforme", "bimodal", "normal_outliers", "ceros"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main() -> int:
    df = pd.read_parquet(RUTA)
    df = df[df["dataset"].isin(SINTETICOS) & df["bins_solicitados"].notna()]
    if df.empty:
        print("No hay filas de sintéticos en", RUTA)
        return 1
    TABLAS.mkdir(parents=True, exist_ok=True)

    prueba = df[df["particion"] == "prueba"]
    tabla = prueba.pivot_table(index=["dataset", "metodo"], columns="bins_solicitados", values="nrmse")
    tabla.to_csv(TABLAS / "resumen_sinteticos_nrmse.csv")
    for ds in SINTETICOS:
        if ds in tabla.index.get_level_values(0):
            print(f"\n=== NRMSE (prueba) · {ds} ===")
            print(tabla.loc[ds].round(4).to_string())

    # Ranking: 1 = menor NRMSE entre los 9 métodos, por dataset y bins; promedio por método
    pr = prueba.copy()
    pr["rank"] = pr.groupby(["dataset", "bins_solicitados"])["nrmse"].rank(method="min")
    ranking = pr.pivot_table(index="metodo", columns="dataset", values="rank", aggfunc="mean")
    ranking["media"] = ranking.mean(axis=1)
    ranking = ranking.sort_values("media")
    ranking.to_csv(TABLAS / "ranking_sinteticos.csv")
    print("\n=== Ranking medio (1 = mejor NRMSE; promedio sobre bins) ===")
    print(ranking.round(2).to_string())

    # Generalización: diferencia relativa de NRMSE entre prueba y ajuste
    aj = df[df["particion"] == "ajuste"].set_index(["dataset", "metodo", "bins_solicitados"])["nrmse"]
    pu = prueba.set_index(["dataset", "metodo", "bins_solicitados"])["nrmse"]
    rel = ((pu - aj) / aj).abs()
    print("\nMáxima diferencia relativa de NRMSE prueba vs ajuste por dataset:")
    print(rel.groupby(level=0).max().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
