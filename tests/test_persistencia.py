import json

import numpy as np
import pytest

from src.cuantizacion import (
    Cuantizador, ajustar_cuantizador, transformar, reconstruir, REPRESENTANTES,
    guardar_modelo, cargar_modelo, guardar_modelos, cargar_modelos,
)


def _modelo(estrategia="cuantiles", k=10, n=20_000, semilla=0):
    x = np.random.default_rng(semilla).lognormal(size=n)
    return x, ajustar_cuantizador(x, estrategia, k, columna="demo")


def _iguales(a, b):
    assert a.estrategia == b.estrategia and a.columna == b.columna
    assert a.n_bins_solicitado == b.n_bins_solicitado
    assert a.n_bins_real == b.n_bins_real and a.n_ajuste == b.n_ajuste
    assert np.array_equal(a.limites, b.limites)
    assert set(a.representantes) == set(b.representantes)
    for r in a.representantes:
        assert np.array_equal(a.representantes[r], b.representantes[r])


def test_to_dict_from_dict():
    _, m = _modelo()
    _iguales(m, Cuantizador.from_dict(m.to_dict()))


@pytest.mark.parametrize("estrategia", ["uniforme", "cuantiles"])
def test_ida_y_vuelta_json(tmp_path, estrategia):
    x, m = _modelo(estrategia, 25)
    ruta = tmp_path / "sub" / "modelo.json"          # la carpeta no existe: se crea
    guardar_modelo(m, ruta)
    m2 = cargar_modelo(ruta)
    _iguales(m, m2)
    assert np.array_equal(transformar(x, m), transformar(x, m2))
    for r in REPRESENTANTES:
        assert np.array_equal(reconstruir(transformar(x, m), m, r),
                              reconstruir(transformar(x, m2), m2, r))


def test_archivo_json_estricto_con_version_y_fecha(tmp_path):
    _, m = _modelo()
    ruta = tmp_path / "m.json"
    guardar_modelo(m, ruta)
    d = json.loads(ruta.read_text(encoding="utf-8"))
    for campo in ("version", "columna", "estrategia", "n_bins_solicitado", "n_bins_real",
                  "limites", "representantes", "n_ajuste", "fecha"):
        assert campo in d


def test_nan_se_guarda_como_null():
    m = Cuantizador("uniforme", 2, 2, np.array([0.0, 1.0, 2.0]),
                    {"punto_medio": np.array([0.5, np.nan])}, "x", 3)
    d = m.to_dict()
    assert d["representantes"]["punto_medio"][1] is None
    json.dumps(d, allow_nan=False)                   # no debe lanzar
    assert np.isnan(Cuantizador.from_dict(d).representantes["punto_medio"][1])


def test_version_incorrecta_y_campos_faltantes():
    _, m = _modelo()
    d = m.to_dict()
    d["version"] = 999
    with pytest.raises(ValueError):
        Cuantizador.from_dict(d)
    d = m.to_dict()
    del d["limites"]
    with pytest.raises(ValueError):
        Cuantizador.from_dict(d)


def test_tamanos_inconsistentes():
    _, m = _modelo()
    d = m.to_dict()
    d["limites"] = d["limites"][:-1]
    with pytest.raises(ValueError):
        Cuantizador.from_dict(d)


def test_guardar_y_cargar_varios_modelos(tmp_path):
    _, m1 = _modelo("uniforme", 5, semilla=1)
    _, m2 = _modelo("cuantiles", 10, semilla=2)
    ruta = tmp_path / "todos.json"
    guardar_modelos({"a": m1, "b": m2}, ruta)
    cargados = cargar_modelos(ruta)
    assert set(cargados) == {"a", "b"}
    _iguales(m1, cargados["a"])
    _iguales(m2, cargados["b"])


def test_transformar_no_modifica_el_modelo():
    x, m = _modelo()
    limites = m.limites.copy()
    reps = {r: v.copy() for r, v in m.representantes.items()}
    nuevo = np.random.default_rng(9).lognormal(mean=1.0, size=5000)   # distribución distinta
    transformar(nuevo, m)
    transformar(nuevo, m, devolver_diagnostico=True)
    assert np.array_equal(limites, m.limites)
    for r in reps:
        assert np.array_equal(reps[r], m.representantes[r])


def test_reutilizacion_con_valores_fuera_de_rango():
    x = np.random.default_rng(0).normal(size=10_000)
    m = ajustar_cuantizador(x[:5000], "uniforme", 10)
    cod, diag = transformar(x[5000:], m, devolver_diagnostico=True)
    assert cod.min() >= 0 and cod.max() <= m.n_bins_real - 1
    assert 0 <= diag["fuera_rango"].mean() < 0.05