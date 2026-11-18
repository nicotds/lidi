"""Línea de comandos del benchmark.

  python scripts/ejecutar_benchmark.py --config benchmark/config.yaml --salida resultados/actividad4/crudo
  (piloto)  ... --salida resultados/actividad4/_piloto --datasets normal lognormal --n-filas 100000 --bins 10 50
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # antes de importar numpy

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from benchmark.runner import avisos_tabla, cargar_config, ejecutar, validar_tabla  # noqa: E402


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description="Ejecuta el benchmark de cuantización.")
    a.add_argument("--config", default=str(RAIZ / "benchmark" / "config.yaml"))
    a.add_argument("--salida", default=str(RAIZ / "resultados" / "actividad4" / "crudo"))
    a.add_argument("--datasets", nargs="+", default=None, help="subconjunto (p. ej. normal lognormal 1M)")
    a.add_argument("--n-filas", type=int, default=None, help="usar solo N filas (piloto)")
    a.add_argument("--bins", nargs="+", type=int, default=None)
    a.add_argument("--metodos", nargs="+", default=None, help="subconjunto de métodos (sin controles)")
    args = a.parse_args(argv)

    cfg = cargar_config(args.config)
    destino = ejecutar(cfg, args.salida, args.datasets, args.n_filas, args.bins, args.metodos)
    df = pd.read_parquet(destino)
    print(f"\n{len(df)} filas -> {destino}")
    avisos = avisos_tabla(df)
    if avisos:
        print(f"AVISOS ({len(avisos)}): el NRMSE sube al subir los bins en (revisar y explicar):")
        for x in avisos[:15]:
            print(" -", x)
    problemas = validar_tabla(df)
    if problemas:
        print("PROBLEMAS DE VALIDACIÓN:")
        for p in problemas:
            print(" -", p)
        return 1
    print("VALIDACIÓN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
