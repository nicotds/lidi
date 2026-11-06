import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "ejecutar_dataset.py"


def test_script_corre_y_valida(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "edad": rng.normal(40, 10, n),
        "ingresos": rng.lognormal(10, 0.5, n),
        "altura": rng.normal(170, 8, n),
    })
    df.loc[rng.choice(n, 100, replace=False), "edad"] = np.nan
    ruta = tmp_path / "mini.parquet"
    df.to_parquet(ruta)

    salida = tmp_path / "salida"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--ruta", str(ruta), "--etiqueta", "mini",
         "--salida", str(salida), "--bins", "5", "10"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    res = pd.read_csv(salida / "tablas" / "resultados_mini.csv")
    assert len(res) == 3 * 2 * 3 * 2              # columnas x estrategias x representantes x bins
    assert res["validacion_ok"].all()
    assert (res["n_filas"] == n).all()
    assert (res["bins_reales"] <= res["bins_solicitados"]).all()
    assert abs(res.loc[res["columna"] == "edad", "pct_nan"].iloc[0] - 2.0) < 1e-9
    for estr in ("uniforme", "cuantiles"):
        for k in (5, 10):
            assert (salida / "modelos" / "mini" / f"{estr}_{k}.json").exists()
            assert (salida / "datos_cuantizados" / "mini" / f"codigos_{estr}_{k}.parquet").exists()