import numpy as np
import pandas as pd
import pytest

from src.cuantizacion import (
    cuantizar_dataframe, reconstruir_dataframe, comparar_estadisticas,
    tabla_resultados,
)

COLS = ["edad", "ingresos", "altura"]


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    d = pd.DataFrame({
        "edad": rng.normal(40, 10, n),
        "categoria": rng.choice(["a", "b", "c"], n),
        "ingresos": rng.lognormal(10, 0.5, n),
        "experiencia": rng.integers(0, 30, n),
        "altura": rng.normal(170, 8, n),
    })
    d.loc[rng.choice(n, 50, replace=False), "edad"] = np.nan
    return d


def test_cuantizar_no_modifica_original(df):
    copia = df.copy()
    cuantizar_dataframe(df, COLS, "uniforme", 10)
    pd.testing.assert_frame_equal(df, copia)


def test_cuantizar_conserva_estructura(df):
    out, modelos = cuantizar_dataframe(df, COLS, "cuantiles", 10)
    otras = [c for c in df.columns if c not in COLS]
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)
    assert set(modelos) == set(COLS)
    for c in COLS:
        assert out[c].dtype == np.int16
    pd.testing.assert_frame_equal(out[otras], df[otras])


def test_nan_se_codifican_como_menos_uno(df):
    out, _ = cuantizar_dataframe(df, COLS, "uniforme", 10)
    assert (out["edad"] == -1).sum() == df["edad"].isna().sum()


def test_reconstruir_dataframe(df):
    out, modelos = cuantizar_dataframe(df, COLS, "cuantiles", 10)
    limites_antes = {c: m.limites.copy() for c, m in modelos.items()}
    rec = reconstruir_dataframe(out, modelos, "media")
    otras = [c for c in df.columns if c not in COLS]

    assert rec.shape == df.shape
    assert list(rec.columns) == list(df.columns)
    pd.testing.assert_frame_equal(rec[otras], df[otras])
    assert np.array_equal(rec["edad"].isna(), df["edad"].isna())
    for c in COLS:                                   # no se recalcularon límites
        assert np.array_equal(modelos[c].limites, limites_antes[c])
        v = rec[c].dropna()
        assert v.min() >= modelos[c].limites[0] and v.max() <= modelos[c].limites[-1]


def test_reconstruir_solo_con_codigos_y_modelos(df):
    _, modelos = cuantizar_dataframe(df, COLS, "uniforme", 5)
    solo_codigos = pd.DataFrame({"edad": np.array([0, 1, -1], dtype=np.int16),
                                 "otra": [1.0, 2.0, 3.0]})
    rec = reconstruir_dataframe(solo_codigos, {"edad": modelos["edad"]})
    assert np.isnan(rec.loc[2, "edad"])
    assert list(rec["otra"]) == [1.0, 2.0, 3.0]


def test_columnas_inexistentes(df):
    with pytest.raises(ValueError):
        cuantizar_dataframe(df, ["no_existe"], "uniforme", 5)
    out, modelos = cuantizar_dataframe(df, COLS, "uniforme", 5)
    with pytest.raises(ValueError):
        reconstruir_dataframe(out.drop(columns=["edad"]), modelos)
    with pytest.raises(ValueError):
        cuantizar_dataframe(df, [], "uniforme", 5)


def test_comparar_estadisticas_identicos():
    o = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [5.0, 3, 4, 1, 2]})
    est = comparar_estadisticas(o, o.copy())
    assert np.allclose(est["resumen"].drop(columns="columna").to_numpy(), 0)
    assert est["dif_correlacion_max"] == pytest.approx(0)


def test_comparar_estadisticas_valores():
    o = pd.DataFrame({"a": [1.0, 2, 3, 4]})
    r = o + 1
    res = comparar_estadisticas(o, r)["resumen"].iloc[0]
    assert res["dif_media"] == pytest.approx(1.0)
    assert res["dif_std"] == pytest.approx(0.0)
    assert res["dif_q50"] == pytest.approx(1.0)


def test_tabla_resultados_filas_y_columnas(df):
    t = tabla_resultados(df, COLS, bins=(5, 10))
    assert len(t) == 3 * 2 * 3 * 2                  # columnas x estrategias x representantes x bins
    for col in ["columna", "estrategia", "representante", "bins_solicitados", "bins_reales",
                "mae", "rmse", "error_max", "dif_media", "dif_std", "dif_q05", "dif_q95",
                "dif_corr_max", "t_ajuste_s", "t_transformar_s", "t_reconstruir_s"]:
        assert col in t.columns
    assert (t["bins_reales"] <= t["bins_solicitados"]).all()
    assert t["mae"].notna().all()