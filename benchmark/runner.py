"""Runner del benchmark.

Un caso es dataset x columna x método x bins. Por cada caso: divide en ajuste/prueba,
ajusta con la mitad A, codifica y decodifica, mide tiempos (1 calentamiento + N
repeticiones, mediana) y memoria pico (tracemalloc), y escribe una fila por partición
("ajuste": evaluado sobre A; "prueba": evaluado sobre la otra mitad B). Los tiempos, la
memoria y la compresión se miden una vez por caso y se repiten en las dos filas.

Los resultados se guardan por dataset en resultados_benchmark_<dataset>.parquet y son
reanudables: si un (método, bins) ya está en el archivo, se salta.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmark import metricas as M
from benchmark.adaptadores import crear_adaptador, es_control
from benchmark.datasets import cargar_real, generar

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS = ["dataset", "columna", "metodo", "estrategia", "representante", "bins_solicitados",
            "bins_reales", "particion", "n_filas", "nmae", "nrmse", "err_max_norm", "w1_norm",
            "ks", "dist_hist", "err_corr", "t_ajuste_s", "t_codificar_s", "t_decodificar_s",
            "mem_pico_mb", "bits_por_valor", "razon_compresion", "pct_fuera_rango", "t_caso_s"]
CLAVE = ["dataset", "columna", "metodo", "bins_solicitados", "particion"]


# ----------------------------------------------------------------------------
# Configuración y utilidades
# ----------------------------------------------------------------------------
def _ruta(r) -> Path:
    """Rutas relativas de la configuración se leen desde la raíz del proyecto."""
    p = Path(r)
    return p if p.is_absolute() else RAIZ / p


def cargar_config(ruta) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def lista_metodos(cfg: dict) -> list[str]:
    """Métodos cuantizadores (propios + sklearn) en orden, sin los controles."""
    m = cfg["metodos"]
    propios = [f"propio_{e}_{r}" for e in m["propios"] for r in m["representantes"]]
    return propios + [f"sklearn_{s}" for s in m["sklearn"]]


def dividir(n: int, frac: float, semilla: int):
    """Índices ordenados de la mitad de ajuste (A) y de prueba (B)."""
    perm = np.random.default_rng(semilla).permutation(n)
    k = int(round(n * frac))
    return np.sort(perm[:k]), np.sort(perm[k:])


def muestra_idx(n: int, tam, semilla: int):
    """Índices de una muestra fija de `tam` filas (None si no aplica)."""
    if not tam or n <= tam:
        return None
    return np.sort(np.random.default_rng(semilla).choice(n, int(tam), replace=False))


def medir(fn, calentamiento: int, repeticiones: int):
    """Devuelve (mediana de segundos, último resultado)."""
    res = None
    for _ in range(calentamiento):
        res = fn()
    ts = []
    for _ in range(repeticiones):
        gc.collect()
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), res


def memoria_pico_mb(fabrica, xa) -> float:
    """Pico de tracemalloc durante ajustar + codificar + decodificar (en MB)."""
    gc.collect()
    ad = fabrica()
    tracemalloc.start()
    try:
        ad.ajustar(xa)
        ad.decodificar(ad.codificar(xa))
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pico / 1e6


def escribir_metadata(salida: Path, cfg: dict) -> Path:
    import numpy, pandas, scipy, sklearn
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        ram_gb = None
    meta = {
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "so": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "nucleos": os.cpu_count(), "ram_total_gb": ram_gb,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "semilla": cfg["semilla"], "config": cfg,
    }
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / "metadata.json"
    destino.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return destino


# ----------------------------------------------------------------------------
# Un caso
# ----------------------------------------------------------------------------
def ejecutar_caso(nombre_ds, columna, xa, xb, metodo, bins, cfg, idx_a, idx_b):
    """Ejecuta un caso. Devuelve ([fila_ajuste, fila_prueba], (ra, rb))."""
    t_inicio = time.perf_counter()
    cal, rep = cfg["tiempos"]["calentamiento"], cfg["tiempos"]["repeticiones"]
    kmax = cfg.get("kmeans_max_ajuste")
    semilla = cfg["semilla"]
    ad = crear_adaptador(metodo, bins, semilla, kmax)

    t_aj, _ = medir(lambda: ad.ajustar(xa), cal, rep)
    t_cod, ca = medir(lambda: ad.codificar(xa), cal, rep)
    t_dec, ra = medir(lambda: ad.decodificar(ca), cal, rep)
    ra = np.asarray(ra, dtype="float64")
    cb = ad.codificar(xb)
    rb = np.asarray(ad.decodificar(cb), dtype="float64")

    mem = (memoria_pico_mb(lambda: crear_adaptador(metodo, bins, semilla, kmax), xa)
           if cfg.get("medir_memoria", True) else float("nan"))

    bins_reales = ad.bins_reales
    bits = ad.bits_por_valor
    base = {"dataset": nombre_ds, "columna": columna, "metodo": metodo,
            "estrategia": ad.estrategia, "representante": ad.representante,
            "bins_solicitados": float("nan") if es_control(metodo) else int(bins),
            "bins_reales": float("nan") if bins_reales is None else int(bins_reales),
            "err_corr": float("nan"),
            "t_ajuste_s": t_aj, "t_codificar_s": t_cod, "t_decodificar_s": t_dec,
            "mem_pico_mb": mem, "bits_por_valor": bits, "razon_compresion": M.razon_compresion(bits)}
    filas = []
    for part, o, r, idx in (("ajuste", xa, ra, idx_a), ("prueba", xb, rb, idx_b)):
        f = dict(base)
        f.update(particion=part, n_filas=int(o.size), pct_fuera_rango=M.pct_fuera_rango(o, ad.limites))
        f.update(M.error_puntual(o, r))
        f.update(M.metricas_distribucion(o, r, idx))
        filas.append(f)
    t_caso = time.perf_counter() - t_inicio
    for f in filas:
        f["t_caso_s"] = t_caso
    return filas, (ra, rb)


# ----------------------------------------------------------------------------
# Un dataset completo
# ----------------------------------------------------------------------------
def _leer_hechos(destino: Path) -> tuple[list[dict], set]:
    if not destino.exists():
        return [], set()
    df = pd.read_parquet(destino)
    hechos = {(m, 0 if pd.isna(b) else int(b)) for m, b in zip(df["metodo"], df["bins_solicitados"])}
    return df.to_dict("records"), hechos


def _guardar(filas: list[dict], destino: Path) -> None:
    tmp = destino.with_suffix(".tmp")
    pd.DataFrame(filas, columns=COLUMNAS).to_parquet(tmp, index=False)
    os.replace(tmp, destino)


def ejecutar_dataset(nombre_ds: str, columnas: dict, cfg: dict, salida: Path, metodos: list[str],
                     bins_lista: list[int], controles: list[str], verbose: bool = True) -> Path:
    """Corre todos los métodos y bins sobre {columna: ndarray}. Reanudable."""
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f"resultados_benchmark_{nombre_ds}.parquet"
    filas, hechos = _leer_hechos(destino)

    nombres = list(columnas)
    n = len(columnas[nombres[0]])
    ia, ib = dividir(n, cfg["particion"]["frac_ajuste"], cfg["particion"]["semilla"])
    tam_m = cfg.get("muestra_distribucion", {}).get(nombre_ds)
    idx_a = muestra_idx(len(ia), tam_m, cfg["semilla"])
    idx_b = muestra_idx(len(ib), tam_m, cfg["semilla"])
    particiones = {c: (columnas[c][ia], columnas[c][ib]) for c in nombres}

    grupos = [(m, b) for b in bins_lista for m in metodos] + [(c, None) for c in controles]
    for i, (metodo, bins) in enumerate(grupos, 1):
        if (metodo, 0 if bins is None else int(bins)) in hechos:
            continue
        t0 = time.perf_counter()
        nuevas, recs = [], {}
        for c in nombres:
            xa, xb = particiones[c]
            f, r = ejecutar_caso(nombre_ds, c, xa, xb, metodo, bins, cfg, idx_a, idx_b)
            nuevas.extend(f)
            recs[c] = r
        if len(nombres) > 1:                         # error de correlación (solo datasets multicolumna)
            for part, orig in (("ajuste", 0), ("prueba", 1)):
                o_df = pd.DataFrame({c: particiones[c][orig] for c in nombres})
                r_df = pd.DataFrame({c: recs[c][orig] for c in nombres})
                err = M.error_correlacion(o_df, r_df)
                for f in nuevas:
                    if f["particion"] == part:
                        f["err_corr"] = err
        filas.extend(nuevas)
        _guardar(filas, destino)
        if verbose:
            print(f"  [{nombre_ds}] {i}/{len(grupos)} {metodo} bins={bins}  {time.perf_counter() - t0:.1f} s", flush=True)
    return destino


def unir_resultados(salida: Path) -> Path:
    """Une los archivos por dataset en resultados_benchmark.parquet."""
    partes = sorted(salida.glob("resultados_benchmark_*.parquet"))
    if not partes:
        raise FileNotFoundError(f"no hay resultados en {salida}")
    df = pd.concat([pd.read_parquet(p) for p in partes], ignore_index=True)
    destino = salida / "resultados_benchmark.parquet"
    df.to_parquet(destino, index=False)
    return destino


def ejecutar(cfg: dict, salida, datasets=None, n_filas=None, bins=None, metodos=None,
             verbose: bool = True) -> Path:
    """Corre el benchmark según la configuración (con sobrescrituras opcionales)."""
    salida = Path(salida)
    escribir_metadata(salida, cfg)
    todos = list(cfg["sinteticos"]["distribuciones"]) + list(cfg["reales"]["datasets"])
    elegidos = list(datasets) if datasets else todos
    bins_lista = list(bins) if bins else list(cfg["bins"])
    metodos_lista = list(metodos) if metodos else lista_metodos(cfg)
    controles = [] if metodos else list(cfg["metodos"]["controles"])

    for nombre in elegidos:
        if nombre in cfg["sinteticos"]["distribuciones"]:
            n = n_filas or cfg["sinteticos"]["n_filas"]
            cols = {"valor": generar(nombre, n, cfg["semilla"], cfg["sinteticos"]["pct_nan"])}
        elif nombre in cfg["reales"]["datasets"]:
            cols = cargar_real(_ruta(cfg["reales"]["datasets"][nombre]), cfg["reales"]["columnas"])
            if n_filas and n_filas < len(next(iter(cols.values()))):
                idx = np.sort(np.random.default_rng(cfg["semilla"]).choice(
                    len(next(iter(cols.values()))), n_filas, replace=False))
                cols = {c: v[idx] for c, v in cols.items()}
        else:
            raise ValueError(f"dataset desconocido: {nombre!r}")
        if verbose:
            print(f"[{nombre}] {len(next(iter(cols.values()))):,} filas, columnas {list(cols)}", flush=True)
        ejecutar_dataset(nombre, cols, cfg, salida, metodos_lista, bins_lista, controles, verbose)
        del cols
    return unir_resultados(salida)


# ----------------------------------------------------------------------------
# Validación de la tabla
# ----------------------------------------------------------------------------
def validar_tabla(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien)."""
    p = []
    if df.duplicated(CLAVE).any():
        p.append(f"{int(df.duplicated(CLAVE).sum())} filas duplicadas en {CLAVE}")
    por = df.groupby(["dataset", "columna"]).size()
    if por.nunique() > 1:
        p.append(f"número de filas distinto entre dataset/columna: {por.to_dict()}")
    if set(df["particion"]) != {"ajuste", "prueba"}:
        p.append(f"particiones inesperadas: {set(df['particion'])}")
    cuant = df[df["bins_solicitados"].notna()]
    if (cuant["bins_reales"] > cuant["bins_solicitados"]).any():
        p.append("hay bins_reales > bins_solicitados")
    for col in ("nmae", "nrmse", "err_max_norm", "w1_norm", "ks", "dist_hist"):
        n = int(df[col].isna().sum())
        if n:
            p.append(f"{n} NaN en {col}")
    for col in ("t_ajuste_s", "t_codificar_s", "t_decodificar_s", "bits_por_valor"):
        if df[col].isna().any() or (df[col] < 0).any():
            p.append(f"valores inválidos en {col}")
    return p


def avisos_tabla(df: pd.DataFrame) -> list[str]:
    """Avisos informativos (no son errores): casos propios donde el NRMSE sube al subir los bins.

    En el piloto con normal/lognormal no debería haber ninguno. En la corrida completa puede
    haberlos de forma legítima (p. ej. cuantiles con 70 % de ceros o uniforme con outliers):
    hay que explicarlos en el informe, no esconderlos.
    """
    prop = df[(df["metodo"].str.startswith("propio_")) & (df["particion"] == "ajuste")]
    avisos = []
    for (ds, c, m), g in prop.groupby(["dataset", "columna", "metodo"]):
        g = g.sort_values("bins_solicitados")
        if (np.diff(g["nrmse"].to_numpy()) > 1e-9).any():
            avisos.append(f"{ds}/{c}/{m}")
    return avisos
