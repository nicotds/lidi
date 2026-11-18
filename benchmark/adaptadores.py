"""Interfaz común de los métodos del benchmark.

Todos implementan ajustar(x), codificar(x) y decodificar(codigos). Los cuantizadores
codifican a int16 con NaN -> -1 y decodifican -1 -> NaN. Los controles (float32 y
sin cuantizar) devuelven valores, no códigos. El runner no distingue entre ellos.

Notas que van al informe:
- Propio: ajustar_cuantizador calcula los tres representantes en cada ajuste, así que
  el tiempo de ajuste de las tres variantes de una estrategia es el mismo.
- Scikit-Learn: se ajusta solo con valores no NaN (los NaN se reinsertan con -1) y con
  subsample=None y quantile_method="linear" para ser comparable con el módulo propio.
  inverse_transform reconstruye con el punto medio de los límites (también en kmeans:
  no usa los centroides).
"""
from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from src.cuantizacion import (ESTRATEGIAS, REPRESENTANTES, ajustar_cuantizador, reconstruir,
                              transformar)

PARAMS_KBINS = inspect.signature(KBinsDiscretizer.__init__).parameters
NOMBRES_PROPIOS = [f"propio_{e}_{r}" for e in ESTRATEGIAS for r in REPRESENTANTES]
NOMBRES_SKLEARN = ["sklearn_uniform", "sklearn_quantile", "sklearn_kmeans"]
NOMBRES_CONTROL = ["float32", "sin_cuantizar"]


class Adaptador:
    nombre = "base"
    estrategia = ""
    representante = ""
    bins_solicitados = None

    def ajustar(self, x):
        raise NotImplementedError

    def codificar(self, x):
        raise NotImplementedError

    def decodificar(self, codigos):
        raise NotImplementedError

    @property
    def bins_reales(self):
        return None

    @property
    def limites(self):
        return None

    @property
    def bits_por_valor(self) -> int:
        """ceil(log2(bins_reales + 1)); el +1 es el código de NaN."""
        return math.ceil(math.log2(self.bins_reales + 1))


class AdaptadorPropio(Adaptador):
    def __init__(self, estrategia: str, representante: str, bins: int):
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"estrategia {estrategia!r} no válida")
        if representante not in REPRESENTANTES:
            raise ValueError(f"representante {representante!r} no válido")
        self.estrategia, self.representante, self.bins_solicitados = estrategia, representante, int(bins)
        self.nombre = f"propio_{estrategia}_{representante}"
        self.modelo = None

    def ajustar(self, x):
        self.modelo = ajustar_cuantizador(x, self.estrategia, self.bins_solicitados)

    def codificar(self, x):
        return transformar(x, self.modelo)

    def decodificar(self, codigos):
        return reconstruir(codigos, self.modelo, self.representante)

    @property
    def bins_reales(self):
        return self.modelo.n_bins_real

    @property
    def limites(self):
        return self.modelo.limites


class AdaptadorSklearn(Adaptador):
    _ESTRATEGIA = {"uniform": "uniforme", "quantile": "cuantiles", "kmeans": "kmeans"}

    def __init__(self, strategy: str, bins: int, semilla: int = 42, max_ajuste: int | None = None):
        if strategy not in self._ESTRATEGIA:
            raise ValueError(f"strategy {strategy!r} no válida")
        self.strategy, self.bins_solicitados = strategy, int(bins)
        self.estrategia = self._ESTRATEGIA[strategy]
        self.representante = "punto_medio_limites"
        self.nombre = f"sklearn_{strategy}"
        self.semilla, self.max_ajuste = semilla, max_ajuste
        self.est = None
        self._constante = None

    def _crear(self) -> KBinsDiscretizer:
        kw = dict(n_bins=self.bins_solicitados, encode="ordinal", strategy=self.strategy)
        if "subsample" in PARAMS_KBINS:
            kw["subsample"] = None
        if self.strategy == "quantile" and "quantile_method" in PARAMS_KBINS:
            kw["quantile_method"] = "linear"
        if "random_state" in PARAMS_KBINS:
            kw["random_state"] = self.semilla
        return KBinsDiscretizer(**kw)

    def ajustar(self, x):
        x = np.asarray(x, dtype="float64")
        v = x[~np.isnan(x)]
        if self.max_ajuste and v.size > self.max_ajuste:
            v = np.random.default_rng(self.semilla).choice(v, self.max_ajuste, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est = self._crear().fit(v.reshape(-1, 1))
        # Columna constante: sklearn usa bordes [-inf, inf] y su inverse_transform da NaN.
        # Se reconstruye con el valor constante (igual que el módulo propio).
        self._constante = None if np.isfinite(self.est.bin_edges_[0]).all() else float(v[0])

    def codificar(self, x):
        x = np.asarray(x, dtype="float64")
        cod = np.full(x.shape, -1, dtype=np.int16)
        m = ~np.isnan(x)
        if m.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cod[m] = self.est.transform(x[m].reshape(-1, 1)).ravel().astype(np.int16)
        return cod

    def decodificar(self, codigos):
        c = np.asarray(codigos)
        out = np.full(c.shape, np.nan)
        m = c >= 0
        if m.any():
            if self._constante is not None:
                out[m] = self._constante
            else:
                out[m] = self.est.inverse_transform(c[m].reshape(-1, 1).astype("float64")).ravel()
        return out

    @property
    def bins_reales(self):
        return len(self.est.bin_edges_[0]) - 1

    @property
    def limites(self):
        if self._constante is not None:
            return np.array([self._constante, self._constante])
        return self.est.bin_edges_[0]


class AdaptadorFloat32(Adaptador):
    nombre = "float32"
    estrategia = "control"
    representante = "float32"

    def ajustar(self, x):
        pass

    def codificar(self, x):
        return np.asarray(x, dtype=np.float32)

    def decodificar(self, codigos):
        return np.asarray(codigos, dtype="float64")

    @property
    def bits_por_valor(self) -> int:
        return 32


class AdaptadorSinCuantizar(Adaptador):
    nombre = "sin_cuantizar"
    estrategia = "control"
    representante = "original"

    def ajustar(self, x):
        pass

    def codificar(self, x):
        return np.array(x, dtype="float64")

    def decodificar(self, codigos):
        return np.asarray(codigos, dtype="float64")

    @property
    def bits_por_valor(self) -> int:
        return 64


def es_control(nombre: str) -> bool:
    return nombre in NOMBRES_CONTROL


def crear_adaptador(nombre: str, bins: int | None = None, semilla: int = 42,
                    kmeans_max_ajuste: int | None = None) -> Adaptador:
    """Crea el adaptador por nombre. Los controles ignoran `bins`."""
    if nombre == "float32":
        return AdaptadorFloat32()
    if nombre == "sin_cuantizar":
        return AdaptadorSinCuantizar()
    if bins is None:
        raise ValueError(f"el método {nombre!r} necesita bins")
    if nombre.startswith("propio_"):
        estrategia, representante = nombre[len("propio_"):].split("_", 1)
        return AdaptadorPropio(estrategia, representante, bins)
    if nombre.startswith("sklearn_"):
        strategy = nombre[len("sklearn_"):]
        return AdaptadorSklearn(strategy, bins, semilla,
                                kmeans_max_ajuste if strategy == "kmeans" else None)
    raise ValueError(f"método desconocido: {nombre!r}")
