"""Cuantización de datos tabulares (código propio).

Convenciones fijadas
--------------------
1. Intervalos semiabiertos [e_i, e_{i+1}); el último incluye el máximo.
   Código = np.searchsorted(limites[1:-1], x, side="right").
2. Valores faltantes (NaN): código -1; al reconstruir vuelven a ser NaN.
3. Fuera de rango: menores al mínimo -> bin 0; mayores al máximo -> último bin.
   Se informa con la máscara booleana `fuera_rango`.
4. Columna constante: un solo bin (n_bins_real = 1), representante = el valor.
   Es la única excepción a "límites estrictamente crecientes": limites = [v, v].
   Nunca hay división por cero.
5. Límites repetidos (cuantiles): se eliminan con np.unique; n_bins_real puede
   ser menor que n_bins_solicitado (p. ej. una columna con 70 % de ceros).
6. Representantes: se guardan los tres (punto_medio, media, mediana).
   Un bin vacío usa el punto medio.
7. Códigos como int16 con signo (por el -1).
8. +-inf se excluyen del ajuste; al transformar, +inf cae en el último bin y
   -inf en el bin 0, ambos marcados como fuera_rango.
9. Columna sin valores finitos: ValueError con mensaje claro.
10. Limitación: los límites uniformes se calculan con np.linspace; con otros
    datos, el redondeo en coma flotante puede mover un valor que cae justo en
    un borde al bin vecino.
11. Cuantiles: límites = np.quantile (interpolación lineal) con el primero
    forzado al mínimo y el último al máximo.
12. calcular_error ignora los pares no finitos (NaN o inf); los NaN deben
    estar en las mismas posiciones en ambos arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ESTRATEGIAS = ("uniforme", "cuantiles")
REPRESENTANTES = ("punto_medio", "media", "mediana")
VERSION_MODELO = 1


@dataclass
class Cuantizador:
    """Modelo de cuantización ajustado sobre una columna."""
    estrategia: str
    n_bins_solicitado: int
    n_bins_real: int
    limites: np.ndarray            # tamaño n_bins_real + 1
    representantes: dict           # {"punto_medio"|"media"|"mediana": array de tamaño n_bins_real}
    columna: str | None
    n_ajuste: int                  # valores finitos usados para ajustar

    def to_dict(self) -> dict:
        """Dict serializable a JSON: arrays a listas, NaN a None, incluye "version"."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "Cuantizador":
        """Inverso de to_dict; devuelve arrays float64 de NumPy."""
        raise NotImplementedError


def _a_float(x) -> np.ndarray:
    """Convierte lista, array o Series a un ndarray float64 unidimensional."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.astype("float64").to_numpy()
    else:
        arr = np.asarray(x, dtype="float64")
    if arr.ndim != 1:
        raise ValueError(f"se esperaba una entrada unidimensional, llegó con {arr.ndim} dimensiones")
    return arr


def _calcular_representantes(validos: np.ndarray, limites: np.ndarray) -> dict:
    """Punto medio, media y mediana de cada bin (vectorizado, sin groupby).

    `validos` son los valores finitos usados en el ajuste. Un bin vacío usa el
    punto medio tanto en la media como en la mediana.
    """
    k = len(limites) - 1
    punto_medio = (limites[:-1] + limites[1:]) / 2.0

    # media por bin con bincount
    codigos = np.searchsorted(limites[1:-1], validos, side="right")
    cuenta = np.bincount(codigos, minlength=k)
    suma = np.bincount(codigos, weights=validos, minlength=k)
    media = np.where(cuenta > 0, suma / np.maximum(cuenta, 1), punto_medio)

    # mediana por bin: ordenar una vez; cada bin es una porción contigua
    ordenado = np.sort(validos)
    inicios = np.searchsorted(ordenado, limites[1:-1], side="left")
    bordes = np.concatenate(([0], inicios, [ordenado.size]))
    mediana = punto_medio.copy()
    for i in range(k):
        if bordes[i + 1] > bordes[i]:
            mediana[i] = np.median(ordenado[bordes[i]:bordes[i + 1]])

    return {"punto_medio": punto_medio, "media": media, "mediana": mediana}


def ajustar_cuantizador(x, estrategia: str = "uniforme", n_bins: int = 10,
                        columna: str | None = None) -> Cuantizador:
    """Ajusta límites y los tres representantes.

    Parámetros: x (lista, array o Series), estrategia (ESTRATEGIAS),
    n_bins (entero >= 1), columna (nombre opcional).
    Retorna: Cuantizador con limites y representantes (punto_medio, media, mediana).
    Excluye NaN y +-inf del ajuste.
    Columna constante: 1 bin con limites = [v, v].
    Cuantiles: los límites repetidos se eliminan, así que n_bins_real puede ser
    menor que n_bins_solicitado.
    Lanza ValueError si n_bins < 1, si la estrategia no existe o si no hay
    valores finitos.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins debe ser un entero >= 1, se recibió {n_bins!r}")
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"estrategia {estrategia!r} no válida; opciones: {ESTRATEGIAS}")

    x = _a_float(x)
    validos = x[np.isfinite(x)]
    if validos.size == 0:
        raise ValueError("la columna no tiene valores finitos")

    vmin, vmax = float(validos.min()), float(validos.max())

    if estrategia == "uniforme":
        if vmin == vmax:                       # columna constante
            limites = np.array([vmin, vmax], dtype="float64")
        else:
            limites = np.linspace(vmin, vmax, int(n_bins) + 1)
    else:                                      # cuantiles
        limites = np.quantile(validos, np.linspace(0.0, 1.0, int(n_bins) + 1))
        limites[0], limites[-1] = vmin, vmax
        limites = np.unique(limites)           # elimina repetidos y ordena
        if limites.size < 2:                   # columna constante
            limites = np.array([vmin, vmax], dtype="float64")

    return Cuantizador(
        estrategia=estrategia,
        n_bins_solicitado=int(n_bins),
        n_bins_real=len(limites) - 1,
        limites=limites,
        representantes=_calcular_representantes(validos, limites),
        columna=columna,
        n_ajuste=int(validos.size),
    )


def transformar(x, modelo: Cuantizador, devolver_diagnostico: bool = False):
    """Asigna códigos int16. NaN -> -1.

    Con devolver_diagnostico=True devuelve (codigos, {"fuera_rango": mascara,
    "es_nan": mascara}). Conserva la longitud de la entrada.
    """
    x = _a_float(x)
    es_nan = np.isnan(x)
    codigos = np.searchsorted(modelo.limites[1:-1], x, side="right")
    codigos[es_nan] = -1                       # searchsorted manda los NaN al último bin
    codigos = codigos.astype(np.int16)

    if not devolver_diagnostico:
        return codigos

    fuera_rango = (x < modelo.limites[0]) | (x > modelo.limites[-1])
    return codigos, {"fuera_rango": fuera_rango, "es_nan": es_nan}


def reconstruir(codigos, modelo: Cuantizador,
                representante: str = "punto_medio") -> np.ndarray:
    """Reemplaza cada código por su representante; -1 -> NaN.

    Parámetros: codigos (enteros en [-1, n_bins_real - 1]), modelo, representante
    ("punto_medio", "media" o "mediana").
    Retorna: ndarray float64 de la misma longitud.
    Lanza ValueError si hay códigos < -1 o > n_bins_real - 1, si no son enteros,
    si el representante no está en REPRESENTANTES o si el modelo no los tiene.
    """
    if representante not in REPRESENTANTES:
        raise ValueError(f"representante {representante!r} no válido; opciones: {REPRESENTANTES}")
    if representante not in modelo.representantes:
        raise ValueError("el modelo no tiene representantes calculados; reajústalo con ajustar_cuantizador")

    c = np.asarray(codigos)
    if c.ndim != 1:
        raise ValueError("los códigos deben ser unidimensionales")
    if c.size and not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"los códigos deben ser enteros, llegaron como {c.dtype}")
    if c.size and (c.min() < -1 or c.max() > modelo.n_bins_real - 1):
        raise ValueError(
            f"códigos fuera de [-1, {modelo.n_bins_real - 1}]: min={c.min()}, max={c.max()}"
        )

    # NaN al final de la tabla: el índice -1 apunta a él sin condicionales
    tabla = np.append(np.asarray(modelo.representantes[representante], dtype="float64"), np.nan)
    return tabla[c.astype(np.int64)]


def calcular_error(original, reconstruido) -> dict:
    """Devuelve {"mae", "rmse", "error_max", "n_validos"}.

    Ignora los pares no finitos. Los NaN deben estar en las mismas posiciones
    en ambos arrays; si no, lanza ValueError. Si no hay pares válidos, las
    métricas son NaN y n_validos = 0.
    """
    o = _a_float(original)
    r = _a_float(reconstruido)
    if o.shape != r.shape:
        raise ValueError(f"longitudes distintas: {o.shape} vs {r.shape}")
    if not np.array_equal(np.isnan(o), np.isnan(r)):
        raise ValueError("los NaN no están en las mismas posiciones")

    validos = np.isfinite(o) & np.isfinite(r)
    n = int(validos.sum())
    if n == 0:
        return {"mae": np.nan, "rmse": np.nan, "error_max": np.nan, "n_validos": 0}

    d = r[validos] - o[validos]
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "error_max": float(np.max(np.abs(d))),
        "n_validos": n,
    }


def cuantizar_dataframe(df: pd.DataFrame, columnas: list, estrategia: str = "uniforme",
                        n_bins: int = 10):
    """Ajusta un cuantizador por columna y las sustituye por códigos int16.

    No modifica df. Retorna (df_codificado, {columna: Cuantizador}).
    Las columnas no listadas pasan intactas.
    """
    raise NotImplementedError


def reconstruir_dataframe(df_codificado: pd.DataFrame, modelos: dict,
                          representante: str = "punto_medio") -> pd.DataFrame:
    """Reconstruye con los modelos guardados, sin recalcular límites.

    Las columnas que no están en `modelos` pasan intactas.
    """
    raise NotImplementedError


def guardar_modelo(modelo: Cuantizador, ruta: str) -> None:
    """Guarda en JSON: version, columna, estrategia, n_bins_solicitado,
    n_bins_real, limites, representantes, n_ajuste y fecha."""
    raise NotImplementedError


def cargar_modelo(ruta: str) -> Cuantizador:
    """Carga un modelo desde JSON. Lanza ValueError si la versión no coincide."""
    raise NotImplementedError


__all__ = [
    "Cuantizador", "ESTRATEGIAS", "REPRESENTANTES",
    "ajustar_cuantizador", "transformar", "reconstruir", "calcular_error",
    "cuantizar_dataframe", "reconstruir_dataframe",
    "guardar_modelo", "cargar_modelo",
]