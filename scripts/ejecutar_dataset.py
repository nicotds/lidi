"""Corrida de cuantización sobre un dataset Parquet.

Ejemplo (desde la raíz del proyecto):
  python scripts/ejecutar_dataset.py --ruta data/dataset_1M.parquet --etiqueta 1M

Salidas (bajo --salida, por defecto resultados/actividad3):
  modelos/<etiqueta>/<estrategia>_<bins>.json          modelos de todas las columnas
  datos_cuantizados/<etiqueta>/codigos_<...>.parquet   códigos int16 (zstd)
  tablas/resultados_<etiqueta>.csv                     una fila por columna x estrategia
                                                       x representante x bins
Código de salida: 0 si todas las validaciones pasan, 1 si alguna falla.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.cuantizacion import (  # noqa: E402
    ESTRATEGIAS, REPRESENTANTES, ajustar_cuantizador, calcular_error,
    comparar_estadisticas, guardar_modelos, reconstruir, transformar,
)

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except ImportError:
    _PROC = None


def rss_mb():
    return round(_PROC.memory_info().rss / 1e6, 1) if _PROC else float("nan")


def parsear(argv=None):
    p = argparse.ArgumentParser(description="Cuantiza un dataset Parquet y registra resultados.")
    p.add_argument("--ruta", required=True, help="archivo .parquet de entrada")
    p.add_argument("--etiqueta", default=None, help="nombre de la corrida (por defecto, el del archivo)")
    p.add_argument("--salida", default=str(RAIZ / "resultados" / "actividad3"))
    p.add_argument("--columnas", nargs="+", default=["edad", "ingresos", "altura"])
    p.add_argument("--bins", nargs="+", type=int, default=[5, 10, 25, 50])
    p.add_argument("--estrategias", nargs="+", choices=ESTRATEGIAS, default=list(ESTRATEGIAS))
    p.add_argument("--muestra", type=int, default=None, help="usar solo N filas al azar")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--guardar-reconstrucciones", action="store_true")
    return p.parse_args(argv)


def ejecutar(a):
    ruta = Path(a.ruta)
    etiqueta = a.etiqueta or ruta.stem
    salida = Path(a.salida)
    d_mod = salida / "modelos" / etiqueta
    d_cod = salida / "datos_cuantizados" / etiqueta
    d_tab = salida / "tablas"
    for d in (d_mod, d_cod, d_tab):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df = pd.read_parquet(ruta, columns=a.columnas)
    t_lectura = time.perf_counter() - t0
    if a.muestra and a.muestra < len(df):
        df = df.sample(n=a.muestra, random_state=a.semilla).reset_index(drop=True)
    n = len(df)
    cols = list(a.columnas)
    datos = {c: df[c].to_numpy(dtype="float64") for c in cols}
    original = pd.DataFrame(datos)
    del df
    print(f"[{etiqueta}] {n:,} filas, columnas {cols}, lectura {t_lectura:.2f} s, RAM {rss_mb()} MB")

    filas = []
    for estr in a.estrategias:
        for k in a.bins:
            tag = f"{estr}_{k}"
            modelos, codigos, valid_cod = {}, {}, {}
            t_aj, t_tr, fuera = {}, {}, {}
            for c in cols:
                ts = time.perf_counter()
                m = ajustar_cuantizador(datos[c], estr, k, columna=c)
                tm = time.perf_counter()
                cod, diag = transformar(datos[c], m, devolver_diagnostico=True)
                te = time.perf_counter()
                modelos[c], codigos[c] = m, cod
                t_aj[c], t_tr[c] = tm - ts, te - tm
                fuera[c] = float(diag["fuera_rango"].mean() * 100)
                valid_cod[c] = bool(
                    len(cod) == n and cod.dtype == np.int16
                    and cod.min() >= -1 and cod.max() <= m.n_bins_real - 1
                    and np.array_equal(cod == -1, np.isnan(datos[c]))
                )

            guardar_modelos(modelos, d_mod / f"{tag}.json")
            ts = time.perf_counter()
            ruta_cod = d_cod / f"codigos_{tag}.parquet"
            pd.DataFrame(codigos).to_parquet(ruta_cod, compression="zstd", index=False)
            t_guardar = time.perf_counter() - ts
            tam_mb = ruta_cod.stat().st_size / 1e6

            for rep in REPRESENTANTES:
                recs, t_rec, ok_rec = {}, {}, {}
                for c in cols:
                    ts = time.perf_counter()
                    recs[c] = reconstruir(codigos[c], modelos[c], rep)
                    t_rec[c] = time.perf_counter() - ts
                    ok_rec[c] = bool(np.array_equal(np.isnan(recs[c]), np.isnan(datos[c])))

                est = comparar_estadisticas(original, pd.DataFrame(recs))
                resumen = est["resumen"].set_index("columna")
                for c in cols:
                    e = calcular_error(datos[c], recs[c])
                    filas.append({
                        "dataset": etiqueta, "n_filas": n, "columna": c,
                        "estrategia": estr, "representante": rep,
                        "bins_solicitados": k, "bins_reales": modelos[c].n_bins_real,
                        "mae": e["mae"], "rmse": e["rmse"], "error_max": e["error_max"],
                        "n_validos": e["n_validos"],
                        "pct_nan": float(np.isnan(datos[c]).mean() * 100),
                        "pct_fuera_rango": fuera[c],
                        **resumen.loc[c].to_dict(),
                        "dif_corr_max": est["dif_correlacion_max"],
                        "t_ajuste_s": t_aj[c], "t_transformar_s": t_tr[c],
                        "t_reconstruir_s": t_rec[c], "t_guardar_codigos_s": t_guardar,
                        "tam_codigos_MB": tam_mb, "rss_MB": rss_mb(),
                        "validacion_ok": valid_cod[c] and ok_rec[c],
                    })
                if a.guardar_reconstrucciones:
                    pd.DataFrame(recs).to_parquet(
                        d_cod / f"reconstruccion_{tag}_{rep}.parquet", compression="zstd", index=False)
                del recs
            print(f"  listo {tag:14s} RAM {rss_mb()} MB")

    tabla = pd.DataFrame(filas)
    destino = d_tab / f"resultados_{etiqueta}.csv"
    tabla.to_csv(destino, index=False)
    todo_ok = bool(tabla["validacion_ok"].all())
    print(f"\nfilas de resultados: {len(tabla)} -> {destino}")
    print("TODAS LAS VALIDACIONES OK" if todo_ok else "HAY VALIDACIONES FALLIDAS (revisar columna validacion_ok)")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(ejecutar(parsear()))