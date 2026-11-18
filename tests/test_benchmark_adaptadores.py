import warnings

import numpy as np
import pandas as pd
import pytest

from benchmark import metricas as M
from benchmark.adaptadores import NOMBRES_CONTROL, NOMBRES_PROPIOS, NOMBRES_SKLEARN, crear_adaptador
from benchmark.datasets import generar
from src.cuantizacion import calcular_error

N = 20_000


# =============== adaptadores ===============
TODOS = NOMBRES_PROPIOS + NOMBRES_SKLEARN + NOMBRES_CONTROL


@pytest.fixture(scope="module")
def datos():
    return generar("lognormal", N, 42)


@pytest.mark.parametrize("nombre", TODOS)
def test_ida_y_vuelta_conserva_forma_y_nan(nombre, datos):
    ad = crear_adaptador(nombre, 10)
    ad.ajustar(datos)
    cod = ad.codificar(datos)
    rec = ad.decodificar(cod)
    assert cod.shape == datos.shape and rec.shape == datos.shape
    assert np.array_equal(np.isnan(rec), np.isnan(datos))
    if nombre not in NOMBRES_CONTROL:
        assert cod.dtype == np.int16
        assert np.array_equal(cod == -1, np.isnan(datos))
        assert ad.bins_reales <= 10 and ad.limites is not None
        v = rec[~np.isnan(rec)]
        assert v.min() >= datos[~np.isnan(datos)].min() - 1e-9 and v.max() <= np.nanmax(datos) + 1e-9
    else:
        assert ad.bins_reales is None and ad.limites is None


def test_sklearn_coincide_con_el_propio_en_punto_medio(datos):
    pares = [("propio_uniforme_punto_medio", "sklearn_uniform"),
             ("propio_cuantiles_punto_medio", "sklearn_quantile")]
    for propio, sk in pares:
        a, b = crear_adaptador(propio, 25), crear_adaptador(sk, 25)
        a.ajustar(datos); b.ajustar(datos)
        assert np.array_equal(a.codificar(datos), b.codificar(datos))
        assert np.allclose(a.decodificar(a.codificar(datos)), b.decodificar(b.codificar(datos)),
                           equal_nan=True)


def test_bits_y_controles(datos):
    ad = crear_adaptador("propio_uniforme_media", 10); ad.ajustar(datos)
    assert ad.bits_por_valor == 4                              # ceil(log2(11))
    ad = crear_adaptador("propio_uniforme_media", 3); ad.ajustar(datos)
    assert ad.bits_por_valor == 2                              # ceil(log2(4))
    assert crear_adaptador("float32").bits_por_valor == 32
    assert crear_adaptador("sin_cuantizar").bits_por_valor == 64
    f32 = crear_adaptador("float32")
    err = M.error_puntual(datos, f32.decodificar(f32.codificar(datos)))
    assert 0 < err["nmae"] < 1e-6
    sc = crear_adaptador("sin_cuantizar")
    assert M.error_puntual(datos, sc.decodificar(sc.codificar(datos)))["mae"] == 0


def test_adaptador_columna_constante():
    x = np.full(1000, 7.0); x[:20] = np.nan
    for nombre in ("propio_cuantiles_media", "sklearn_uniform", "sklearn_quantile", "sklearn_kmeans"):
        ad = crear_adaptador(nombre, 10); ad.ajustar(x)
        rec = ad.decodificar(ad.codificar(x))
        assert np.allclose(rec[20:], 7.0) and np.isnan(rec[:20]).all()
        assert ad.bins_reales == 1


def test_crear_adaptador_invalidos():
    for malo in ("inventado", "propio_uniforme_moda", "sklearn_otro"):
        with pytest.raises(ValueError):
            crear_adaptador(malo, 10)
    with pytest.raises(ValueError):
        crear_adaptador("propio_uniforme_media")             # falta bins


# =============== métricas ===============
X8 = [10, 12, 15, 21, 24, 31, 35, 40]
R8 = [13.75, 13.75, 13.75, 21.25, 21.25, 28.75, 36.25, 36.25]


def test_ejemplo_manual_de_la_actividad_3():
    e = M.error_puntual(X8, R8)
    assert e["mae"] == pytest.approx(2.125)
    assert e["rmse"] == pytest.approx(2.4238, abs=1e-3)
    assert e["err_max"] == pytest.approx(3.75)
    assert e["nmae"] == pytest.approx(2.125 / 30)
    assert e["nrmse"] == pytest.approx(e["rmse"] / np.std(X8))
    assert e["err_max_norm"] == pytest.approx(3.75 / 30)
    ref = calcular_error(X8, R8)                              # coincide con src/cuantizacion.py
    assert e["mae"] == pytest.approx(ref["mae"]) and e["rmse"] == pytest.approx(ref["rmse"])


def test_un_dato_contra_si_mismo(datos):
    d = M.metricas_distribucion(datos, datos)
    assert d == {"w1_norm": 0.0, "ks": 0.0, "dist_hist": 0.0}
    e = M.error_puntual(datos, datos)
    assert e["mae"] == 0 and e["nrmse"] == 0


def test_columna_constante_no_divide_por_cero():
    x = np.full(100, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = M.error_puntual(x, x)
        d = M.metricas_distribucion(x, x)
    assert e["mae"] == 0 and np.isnan(e["nmae"]) and np.isnan(e["nrmse"]) and np.isnan(e["err_max_norm"])
    assert np.isnan(d["w1_norm"]) and np.isnan(d["dist_hist"])


def test_los_nan_se_ignoran():
    o = np.array([1.0, np.nan, 3.0, 5.0]); r = np.array([2.0, np.nan, 3.0, 4.0])
    assert M.error_puntual(o, r)["mae"] == pytest.approx(2 / 3)
    assert np.isnan(M.error_puntual([np.nan], [np.nan])["mae"])
    with pytest.raises(ValueError):
        M.error_puntual([1, 2], [1, 2, 3])


def test_distribucion_detecta_diferencias(datos):
    desplazado = datos * 1.5
    d = M.metricas_distribucion(datos, desplazado)
    assert d["w1_norm"] > 0.1 and d["ks"] > 0.1 and d["dist_hist"] > 0.1


def test_error_de_correlacion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    o = pd.DataFrame({"a": a, "b": a + rng.normal(size=5000), "c": rng.normal(size=5000)})
    assert M.error_correlacion(o, o) == pytest.approx(0)
    r = o.copy(); r["b"] = rng.normal(size=5000)              # destruye la correlación a-b
    assert M.error_correlacion(o, r) > 0.5
    assert np.isnan(M.error_correlacion(o[["a"]], o[["a"]]))


def test_bits_compresion_y_fuera_de_rango():
    assert M.bits_por_valor(255) == 8 and M.bits_por_valor(256) == 9 and M.bits_por_valor(1) == 1
    assert M.razon_compresion(8) == 8
    assert M.pct_fuera_rango([0, 5, 10, 20, np.nan], np.array([1.0, 10.0])) == pytest.approx(50)
    assert M.pct_fuera_rango([1, 2], None) == 0
