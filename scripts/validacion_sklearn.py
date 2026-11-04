"""Validación de src.cuantizacion frente a KBinsDiscretizer (scikit-learn).

Ejecutar desde la raíz del proyecto:  python scripts/validacion_sklearn.py
"""
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import KBinsDiscretizer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.cuantizacion import ajustar_cuantizador, transformar  # noqa: E402

BINS = [5, 10, 25, 50]
PARAMS = inspect.signature(KBinsDiscretizer.__init__).parameters
SALIDA = RAIZ / "resultados" / "actividad3" / "tablas" / "validacion_sklearn.csv"
SALIDA.parent.mkdir(parents=True, exist_ok=True)


def sk_ajustar(x, strategy, k, defecto=False):
    """Ajusta KBinsDiscretizer sobre un vector sin NaN. Devuelve (bordes, códigos, avisos)."""
    kw = dict(n_bins=k, encode="ordinal", strategy=strategy)
    if not defecto:
        if "subsample" in PARAMS:
            kw["subsample"] = None
        if strategy == "quantile" and "quantile_method" in PARAMS:
            kw["quantile_method"] = "linear"
    est = KBinsDiscretizer(**kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est.fit(x.reshape(-1, 1))
        cod = est.transform(x.reshape(-1, 1)).ravel().astype(np.int64)
    avisos = "; ".join(sorted({str(a.message)[:90] for a in w}))
    return est.bin_edges_[0], cod, avisos


def nota_nan(x, strategy, k=10):
    """Comportamiento de sklearn si se le pasan NaN directamente."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            KBinsDiscretizer(n_bins=k, encode="ordinal", strategy=strategy).fit_transform(x.reshape(-1, 1))
        return "sklearn acepta NaN en la entrada"
    except Exception as e:  # noqa: BLE001
        return f"sklearn rechaza NaN ({type(e).__name__})"


def comparar(caso, x, estrategia, k, defecto=False):
    strategy = "uniform" if estrategia == "uniforme" else "quantile"
    validos = x[~np.isnan(x)]                       # la comparación se hace sobre valores válidos
    lim_sk, cod_sk, avisos = sk_ajustar(validos, strategy, k, defecto)
    m = ajustar_cuantizador(validos, estrategia, k)
    cod = transformar(validos, m).astype(np.int64)

    constante = validos.min() == validos.max()
    pct = float(np.mean(cod == cod_sk) * 100)

    if constante:
        iguales, dif = False, np.nan
        causa = "columna constante: sklearn usa bordes [-inf, inf]; el módulo propio usa [v, v]"
    elif len(lim_sk) != len(m.limites):
        iguales, dif = False, np.nan
        causa = (f"distinto número de bins: sklearn elimina bins de ancho <= 1e-8 "
                 f"(sklearn {len(lim_sk) - 1}, propio {m.n_bins_real})")
    else:
        dif = float(np.max(np.abs(lim_sk - m.limites)))
        iguales = bool(np.allclose(lim_sk, m.limites))
        if iguales and pct == 100:
            causa = "ninguna"
        elif defecto:
            causa = "sklearn con parámetros por defecto: submuestreo y/o método de cuantil distinto"
        else:
            causa = "REVISAR: diferencia numérica en límites o en la convención de bordes"

    return {
        "caso": caso, "estrategia": estrategia, "bins": k,
        "config_sklearn": "por_defecto" if defecto else "ajustada",
        "bins_reales_sklearn": len(lim_sk) - 1, "bins_reales_propio": m.n_bins_real,
        "limites_iguales": "sí" if iguales else "no",
        "dif_max_limites": dif, "pct_codigos_iguales": round(pct, 4),
        "causa": causa, "avisos_sklearn": avisos,
        "nota_nan": nota_nan(x, strategy) if np.isnan(x).any() else "",
    }


def construir_casos():
    df = pd.read_parquet(RAIZ / "data" / "dataset_1M.parquet", columns=["edad", "ingresos", "altura"])
    casos = {f"real_{c}": df[c].to_numpy() for c in df.columns}
    rng = np.random.default_rng(0)
    casos["muchos_ceros"] = rng.permutation(
        np.concatenate([np.zeros(70_000), rng.exponential(size=30_000) + 0.1]))
    casos["constante"] = np.full(1_000, 7.0)
    x = rng.normal(50, 10, 100_000)
    x[rng.choice(x.size, 5_000, replace=False)] = np.nan
    casos["con_nan"] = x
    return casos


if __name__ == "__main__":
    print("scikit-learn", sklearn.__version__)
    casos = construir_casos()
    filas = []
    for nombre, x in casos.items():
        for estrategia in ("uniforme", "cuantiles"):
            for k in BINS:
                filas.append(comparar(nombre, x, estrategia, k))
        print("listo:", nombre)
    # Cómo se comporta sklearn con sus parámetros por defecto (1M de filas)
    for nombre in ("real_edad", "real_ingresos", "real_altura"):
        filas.append(comparar(nombre, casos[nombre], "cuantiles", 10, defecto=True))

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(tabla[["caso", "estrategia", "bins", "config_sklearn", "limites_iguales",
                 "dif_max_limites", "pct_codigos_iguales", "causa"]].to_string())
    print("\nguardado en", SALIDA)