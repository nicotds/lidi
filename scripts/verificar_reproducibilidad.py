"""Recarga los modelos JSON y comprueba que reproducen exactamente los códigos guardados.

Ejecutar desde la raíz:  python scripts/verificar_reproducibilidad.py
Código de salida: 0 si todo coincide, 1 si hay diferencias.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.cuantizacion import ESTRATEGIAS, cargar_modelos, transformar  # noqa: E402

COLS = ["edad", "ingresos", "altura"]
BINS = [5, 10, 25, 50]
BASE = RAIZ / "resultados" / "actividad3"


def verificar(etiqueta: str, ruta_datos: Path) -> int:
    df = pd.read_parquet(ruta_datos, columns=COLS)
    fallos = 0
    for estr in ESTRATEGIAS:
        for k in BINS:
            modelos = cargar_modelos(BASE / "modelos" / etiqueta / f"{estr}_{k}.json")
            guardados = pd.read_parquet(
                BASE / "datos_cuantizados" / etiqueta / f"codigos_{estr}_{k}.parquet")
            for c in COLS:
                if not np.array_equal(transformar(df[c], modelos[c]), guardados[c].to_numpy()):
                    fallos += 1
                    print("FALLA:", etiqueta, estr, k, c)
    print(f"{etiqueta}: {'OK (8 modelos x 3 columnas)' if fallos == 0 else f'{fallos} fallos'}")
    return fallos


if __name__ == "__main__":
    total = verificar("1M", RAIZ / "data" / "dataset_1M.parquet")
    total += verificar("5M", RAIZ / "data" / "dataset_5M.parquet")
    sys.exit(1 if total else 0)