import warnings

import numpy as np
import pandas as pd
import pytest

from src.cuantizacion import ajustar_cuantizador, transformar

X = [10, 12, 15, 21, 24, 31, 35, 40]


# ---------- humo ----------
def test_importa_modulo():
    from src import cuantizacion as c
    for nombre in c.__all__:
        assert hasattr(c, nombre)


# ---------- 22/10: ejemplo base ----------
def test_limites_uniformes_ejemplo():
    m = ajustar_cuantizador(X, "uniforme", 4)
    assert np.allclose(m.limites, [10, 17.5, 25, 32.5, 40])
    assert m.n_bins_real == 4
    assert m.n_ajuste == 8


def test_codigos_ejemplo():
    m = ajustar_cuantizador(X, "uniforme", 4)
    codigos = transformar(X, m)
    assert np.array_equal(codigos, [0, 0, 0, 1, 1, 2, 3, 3])
    assert codigos.dtype == np.int16


# ---------- 23/10: bordes ----------
@pytest.mark.parametrize("valor, esperado", [(10, 0), (17.5, 1), (25, 2), (32.5, 3), (40, 3)])
def test_bordes(valor, esperado):
    m = ajustar_cuantizador(X, "uniforme", 4)
    assert transformar([valor], m)[0] == esperado


def test_limites_estrictamente_crecientes():
    m = ajustar_cuantizador(X, "uniforme", 4)
    assert np.all(np.diff(m.limites) > 0)
    assert len(m.limites) == m.n_bins_real + 1


# ---------- 23/10: fuera de rango ----------
def test_fuera_de_rango():
    m = ajustar_cuantizador(X, "uniforme", 4)
    cod, diag = transformar([-5, 10, 40, 100, np.nan], m, devolver_diagnostico=True)
    assert list(cod) == [0, 0, 3, 3, -1]
    assert list(diag["fuera_rango"]) == [True, False, False, True, False]
    assert list(diag["es_nan"]) == [False, False, False, False, True]


# ---------- 23/10: casos especiales ----------
def test_columna_constante():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # cualquier warning hace fallar el test
        m = ajustar_cuantizador([7, 7, 7], "uniforme", 5)
        cod = transformar([7, 7, 7], m)
    assert m.n_bins_real == 1
    assert list(cod) == [0, 0, 0]


def test_toda_nan():
    with pytest.raises(ValueError):
        ajustar_cuantizador([np.nan, np.nan], "uniforme", 3)


def test_solo_infinitos():
    with pytest.raises(ValueError):
        ajustar_cuantizador([np.inf, -np.inf], "uniforme", 3)


def test_valores_repetidos():
    m = ajustar_cuantizador([5, 5, 5, 5, 9], "uniforme", 2)
    assert m.n_bins_real == 2
    assert list(transformar([5, 5, 5, 5, 9], m)) == [0, 0, 0, 0, 1]


def test_infinitos_se_excluyen_del_ajuste():
    x = [1, 2, np.inf, -np.inf, 3, np.nan]
    m = ajustar_cuantizador(x, "uniforme", 2)
    assert m.n_ajuste == 3
    assert np.allclose(m.limites, [1, 2, 3])
    cod, diag = transformar(x, m, devolver_diagnostico=True)
    assert list(cod) == [0, 1, 1, 0, 1, -1]
    assert list(diag["fuera_rango"]) == [False, False, True, True, False, False]


def test_enteros_y_tipos_de_entrada_equivalentes():
    lista = [1, 2, 3, 4, 5, 6, 7, 8]
    entradas = [
        lista,
        np.array(lista),                         # int64
        np.array(lista, dtype=float),
        pd.Series(lista),
        pd.Series(lista, dtype="Int64"),         # entero nulable de pandas
    ]
    referencia = transformar(lista, ajustar_cuantizador(lista, "uniforme", 4))
    for entrada in entradas:
        m = ajustar_cuantizador(entrada, "uniforme", 4)
        assert np.array_equal(transformar(entrada, m), referencia)


def test_n_bins_invalido():
    for k in (0, -3):
        with pytest.raises(ValueError):
            ajustar_cuantizador(X, "uniforme", k)


def test_estrategia_invalida():
    with pytest.raises(ValueError):
        ajustar_cuantizador(X, "inventada", 4)


def test_entrada_no_unidimensional():
    with pytest.raises(ValueError):
        ajustar_cuantizador(np.ones((3, 3)), "uniforme", 2)


# ---------- 23/10: propiedades ----------
@pytest.mark.parametrize("semilla", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 2, 5, 10, 50])
def test_propiedades(semilla, k):
    rng = np.random.default_rng(semilla)
    x = rng.normal(size=1000)
    x[rng.choice(1000, 20, replace=False)] = np.nan
    m = ajustar_cuantizador(x, "uniforme", k)
    cod = transformar(x, m)

    assert len(cod) == len(x)                         # conserva la longitud
    assert cod.dtype == np.int16
    assert cod.min() >= -1 and cod.max() <= m.n_bins_real - 1
    assert np.array_equal(cod == -1, np.isnan(x))     # -1 solo donde hay NaN

    validos = ~np.isnan(x)
    orden = np.argsort(x[validos], kind="stable")
    assert np.all(np.diff(cod[validos][orden]) >= 0)  # monotonía