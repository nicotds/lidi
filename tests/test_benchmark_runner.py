import copy
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import metricas as M
from benchmark.datasets import generar
from benchmark.runner import (avisos_tabla, cargar_config, dividir, ejecutar, lista_metodos,
                              muestra_idx, validar_tabla)

RAIZ = Path(__file__).resolve().parents[1]
CFG = cargar_config(RAIZ / "benchmark" / "config.yaml")


# =============== runner ===============
def _cfg_mini(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["tiempos"] = {"calentamiento": 0, "repeticiones": 1}
    cfg["muestra_distribucion"] = {"mini": 1000}
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.normal(40, 10, n)
    pd.DataFrame({"edad": a, "ingresos": rng.lognormal(10, 0.5, n) + a, "altura": rng.normal(170, 8, n)}
                 ).to_parquet(tmp_path / "mini.parquet")
    cfg["reales"]["datasets"] = {"mini": str(tmp_path / "mini.parquet")}
    cfg["sinteticos"]["distribuciones"] = ["normal", "ceros"]
    return cfg


def test_conteo_de_casos_del_plan():
    assert len(lista_metodos(CFG)) == 9
    assert len(lista_metodos(CFG)) * len(CFG["bins"]) + len(CFG["metodos"]["controles"]) == 56


def test_dividir_es_disjunta_y_reproducible():
    a, b = dividir(1001, 0.5, 42)
    assert len(a) + len(b) == 1001 and not set(a) & set(b)
    assert np.array_equal(np.sort(np.r_[a, b]), np.arange(1001))
    assert np.array_equal(a, dividir(1001, 0.5, 42)[0])


def test_runner_completo_validacion_y_reanudacion(tmp_path):
    cfg = _cfg_mini(tmp_path)
    salida = tmp_path / "salida"
    destino = ejecutar(cfg, salida, datasets=["normal", "ceros", "mini"], n_filas=4000,
                       bins=[5, 10], verbose=False)
    df = pd.read_parquet(destino)
    por_col = (9 * 2 + 2) * 2                                    # métodos x bins + controles, x 2 particiones
    assert len(df) == por_col * (1 + 1 + 3)                      # 2 sintéticos (1 col) + real (3 cols)
    assert validar_tabla(df) == []
    assert (salida / "metadata.json").exists()
    assert set(df["particion"]) == {"ajuste", "prueba"}
    real = df[df["dataset"] == "mini"]
    assert real["err_corr"].notna().all() and df[df["dataset"] == "normal"]["err_corr"].isna().all()
    f32 = df[df["metodo"] == "float32"]
    assert f32["bins_solicitados"].isna().all() and (f32["bits_por_valor"] == 32).all()
    ceros = df[(df["dataset"] == "ceros") & (df["metodo"] == "propio_cuantiles_media")]
    assert (ceros["bins_reales"] < ceros["bins_solicitados"]).all()           # 70 % de ceros colapsa bins

    # reanudación: una segunda corrida no recalcula nada
    mtime = (salida / "resultados_benchmark_normal.parquet").stat().st_mtime_ns
    ejecutar(cfg, salida, datasets=["normal"], n_filas=4000, bins=[5, 10], verbose=False)
    assert (salida / "resultados_benchmark_normal.parquet").stat().st_mtime_ns == mtime


def test_validar_tabla_detecta_problemas(tmp_path):
    cfg = _cfg_mini(tmp_path)
    df = pd.read_parquet(ejecutar(cfg, tmp_path / "s", datasets=["normal"], n_filas=3000,
                                  bins=[5, 10], verbose=False))
    assert validar_tabla(df) == []
    assert any("duplicadas" in p for p in validar_tabla(pd.concat([df, df.iloc[:1]])))
    malo = df.copy(); malo.loc[malo.index[0], "nmae"] = np.nan
    assert any("NaN en nmae" in p for p in validar_tabla(malo))
    malo = df.copy(); malo.loc[malo["bins_solicitados"] == 5, "bins_reales"] = 99
    assert any("bins_reales" in p for p in validar_tabla(malo))


def test_avisos_de_monotonia(tmp_path):
    cfg = _cfg_mini(tmp_path)
    df = pd.read_parquet(ejecutar(cfg, tmp_path / "s2", datasets=["normal"], n_filas=3000,
                                  bins=[5, 10, 50], metodos=["propio_uniforme_media"], verbose=False))
    assert avisos_tabla(df) == []                                # el error baja al subir los bins
    malo = df.copy()
    malo.loc[(malo["particion"] == "ajuste") & (malo["bins_solicitados"] == 50), "nrmse"] = 9.0
    assert avisos_tabla(malo) == ["normal/valor/propio_uniforme_media"]


def test_muestra_fija_para_distribucion():
    datos = generar("lognormal", 20_000, 42)
    idx = muestra_idx(len(datos), 5000, 42)
    assert len(idx) == 5000 and np.array_equal(idx, muestra_idx(len(datos), 5000, 42))
    assert muestra_idx(100, 5000, 42) is None and muestra_idx(100, None, 42) is None
    d = M.metricas_distribucion(datos, datos * 1.1, idx)
    assert d["ks"] > 0
