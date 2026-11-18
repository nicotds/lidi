import numpy as np
import pandas as pd
import pytest

from benchmark.datasets import SINTETICOS, generar, perfil_sinteticos

N = 20_000


@pytest.mark.parametrize("nombre", SINTETICOS)
def test_misma_semilla_mismos_datos(nombre):
    a, b = generar(nombre, N, 42), generar(nombre, N, 42)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, generar(nombre, N, 7), equal_nan=True)


@pytest.mark.parametrize("nombre", SINTETICOS)
def test_dos_por_ciento_de_nan_y_tamano(nombre):
    x = generar(nombre, N, 42)
    assert x.shape == (N,) and x.dtype == np.float64
    assert abs(np.isnan(x).mean() - 0.02) < 1e-9


def test_formas_de_las_distribuciones():
    s = lambda n: pd.Series(generar(n, N, 42))
    assert s("lognormal").skew() > 1.0                       # sesgada a la derecha
    assert abs(s("normal").skew()) < 0.1
    assert abs(s("normal").mean() - 50) < 0.5 and abs(s("normal").std() - 10) < 0.5
    assert s("uniforme").min() >= 0 and s("uniforme").max() <= 100
    bi = s("bimodal").dropna()
    assert ((bi > 45) & (bi < 55)).mean() < 0.01             # hueco entre las dos modas
    assert (bi < 50).mean() == pytest.approx(0.5, abs=0.03)
    ou = s("normal_outliers").dropna()
    assert (ou - 50).abs().max() >= 400                       # 50 desviaciones
    assert ((ou - 50).abs() > 100).mean() == pytest.approx(0.001, abs=0.0006)
    ce = s("ceros").dropna()
    assert (ce == 0).mean() == pytest.approx(0.7, abs=0.02)
    assert (ce[ce != 0] > 0).all()


def test_distribucion_invalida():
    with pytest.raises(ValueError):
        generar("inventada", 100)


def test_perfil_sinteticos():
    t = perfil_sinteticos(n=5000)
    assert list(t["dataset"]) == list(SINTETICOS)
    assert {"media", "desv", "minimo", "maximo", "asimetria"} <= set(t.columns)
