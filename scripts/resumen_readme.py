"""Imprime en Markdown tablas resumen de resultados_actividad3.csv para pegar en el README."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
r = pd.read_csv(RAIZ / "resultados" / "actividad3" / "tablas" / "resultados_actividad3.csv")


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for fila in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in fila) + " |")
    return "\n".join(out)


media = r[r["representante"] == "media"]

print("#### MAE con representante `media`\n")
mae = media.pivot_table(index=["dataset", "columna", "estrategia"],
                        columns="bins_solicitados", values="mae").reset_index()
mae.columns = [str(c) for c in mae.columns]
print(md(mae))

print("\n#### Error máximo con representante `media`\n")
emax = media.pivot_table(index=["dataset", "columna", "estrategia"],
                         columns="bins_solicitados", values="error_max").reset_index()
emax.columns = [str(c) for c in emax.columns]
print(md(emax))

print("\n#### Tiempos (s, suma de las 3 columnas), ajuste y transformación\n")
unica = r.drop_duplicates(["dataset", "columna", "estrategia", "bins_solicitados"])
t = unica.groupby(["dataset", "estrategia", "bins_solicitados"])[
    ["t_ajuste_s", "t_transformar_s"]].sum().reset_index()
print(md(t))

print("\n#### Memoria y tamaño de códigos\n")
mem = r.groupby("dataset").agg(rss_max_MB=("rss_MB", "max"), codigos_MB_max=("tam_codigos_MB", "max")).reset_index()
print(md(mem))

print("\nBins reales distintos de los solicitados:",
      int((r["bins_reales"] != r["bins_solicitados"]).sum()), "filas")