import numpy as np
import pandas as pd
import pytest

from src.cuantizacion import (
    Cuantizador, ajustar_cuantizador, transformar, reconstruir, calcular_error,
    ESTRATEGIAS, REPRESENTANTES,
)

X = list(range(1, 101))


# ---------- n_bins = 1 ----------
@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_un_solo_bin(estrategia):
    m = ajustar_cuantizador(X, estrategia, 1)
    assert m.n_bins_real == 1
    assert np.allclose(m.limites, [1, 100])
    assert np.all(transformar(X, m) == 0)
    for rep in REPRESENTANTES:
        rec = reconstruir(transformar(X, m), m, rep)
        assert np.allclose(rec, 50.5)


# ---------- n_bins mayor que los valores distintos ----------
@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_mas_bins_que_valores_distintos(estrategia):
    x = [1, 1, 2, 2, 3, 3]
    m = ajustar_cuantizador(x, estrategia, 20)
    assert 1 <= m.n_bins_real <= 20
    assert np.all(np.diff(m.limites) > 0)
    cod = transformar(x, m)
    assert cod.min() >= 0 and cod.max() <= m.n_bins_real - 1
    for rep in REPRESENTANTES:
        rec = reconstruir(cod, m, rep)
        assert not np.any(np.isnan(rec))
        assert np.isfinite(calcular_error(x, rec)["mae"])


def test_uniforme_conserva_todos_los_bins_aunque_esten_vacios():
    m = ajustar_cuantizador([1, 1, 2, 2, 3, 3], "uniforme", 20)
    assert m.n_bins_real == 20


# ---------- n_bins inválido ----------
@pytest.mark.parametrize("k", [0, -1, 2.5, "3", None, True, 40_000])
def test_n_bins_invalido(k):
    with pytest.raises(ValueError):
        ajustar_cuantizador(X, "uniforme", k)


def test_n_bins_entero_de_numpy_es_valido():
    assert ajustar_cuantizador(X, "uniforme", np.int64(4)).n_bins_real == 4


# ---------- entrada vacía ----------
@pytest.mark.parametrize("vacia", [[], np.array([]), pd.Series([], dtype="float64")])
def test_ajustar_con_entrada_vacia(vacia):
    with pytest.raises(ValueError):
        ajustar_cuantizador(vacia, "uniforme", 3)


def test_transformar_y_reconstruir_vacios():
    m = ajustar_cuantizador(X, "uniforme", 4)
    cod = transformar([], m)
    assert cod.shape == (0,) and cod.dtype == np.int16
    assert reconstruir(np.array([], dtype=np.int16), m).shape == (0,)
    e = calcular_error([], [])
    assert e["n_validos"] == 0 and np.isnan(e["mae"])


# ---------- infinitos ----------
def test_infinitos_en_transformar_y_reconstruir():
    m = ajustar_cuantizador(X, "cuantiles", 4)
    cod, diag = transformar([np.inf, -np.inf, 50], m, devolver_diagnostico=True)
    assert list(cod) == [m.n_bins_real - 1, 0, cod[2]]
    assert list(diag["fuera_rango"]) == [True, True, False]
    assert np.all(np.isfinite(reconstruir(cod, m)))


def test_calcular_error_ignora_infinitos():
    e = calcular_error([1.0, np.inf], [1.0, 5.0])
    assert e["n_validos"] == 1 and e["mae"] == 0


# ---------- códigos inválidos en reconstruir ----------
@pytest.mark.parametrize("malos", [
    np.array([-2], dtype=np.int16),
    np.array([4], dtype=np.int16),
    np.zeros((2, 2), dtype=np.int16),
    np.array([0.5]),
    np.array([True]),
])
def test_codigos_invalidos(malos):
    m = ajustar_cuantizador(X, "uniforme", 4)
    with pytest.raises(ValueError):
        reconstruir(malos, m)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8])
def test_codigos_enteros_de_cualquier_tipo(dtype):
    m = ajustar_cuantizador(X, "uniforme", 4)
    rec = reconstruir(np.array([0, 1, 3], dtype=dtype), m)
    assert np.allclose(rec, m.representantes["punto_medio"][[0, 1, 3]])


def test_modelo_sin_representantes():
    m = Cuantizador("uniforme", 2, 2, np.array([0.0, 1.0, 2.0]), {}, "x", 3)
    with pytest.raises(ValueError):
        reconstruir(np.array([0, 1], dtype=np.int16), m)