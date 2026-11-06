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
13. DataFrames: las columnas cuantizadas pasan a códigos int16; el resto pasa
    intacto. Las funciones no modifican el DataFrame de entrada.
14. Persistencia: JSON estricto (NaN -> null) con campo "version". Cargar un
    modelo no recalcula nada: límites y representantes se leen tal cual.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ESTRATEGIAS = ("uniforme", "cuantiles")
REPRESENTANTES = ("punto_medio", "media", "mediana")
CUANTILES_ESTADISTICAS = (0.05, 0.25, 0.5, 0.75, 0.95)
VERSION_MODELO = 1


def _lista(a) -> list:
    """ndarray -> lista de floats; NaN/inf -> None (JSON estricto)."""
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype="float64")]


def _a_array(lista) -> np.ndarray:
    """Lista con None -> ndarray float64 con NaN."""
    return np.array([np.nan if v is None else v for v in lista], dtype="float64")


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
        return {
            "version": VERSION_MODELO,
            "columna": self.columna,
            "estrategia": self.estrategia,
            "n_bins_solicitado": int(self.n_bins_solicitado),
            "n_bins_real": int(self.n_bins_real),
            "limites": _lista(self.limites),
            "representantes": {k: _lista(v) for k, v in self.representantes.items()},
            "n_ajuste": int(self.n_ajuste),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cuantizador":
        """Inverso de to_dict; devuelve arrays float64 de NumPy.

        Lanza ValueError si faltan campos, si la versión no coincide o si los
        tamaños no son consistentes.
        """
        requeridos = ("version", "columna", "estrategia", "n_bins_solicitado",
                      "n_bins_real", "limites", "representantes", "n_ajuste")
        faltan = [c for c in requeridos if c not in d]
        if faltan:
            raise ValueError(f"modelo incompleto, faltan campos: {faltan}")
        if d["version"] != VERSION_MODELO:
            raise ValueError(f"versión de modelo {d['version']!r} no soportada (se esperaba {VERSION_MODELO})")

        limites = _a_array(d["limites"])
        representantes = {k: _a_array(v) for k, v in d["representantes"].items()}
        n_real = int(d["n_bins_real"])
        if limites.size != n_real + 1:
            raise ValueError(f"se esperaban {n_real + 1} límites y hay {limites.size}")
        for nombre, r in representantes.items():
            if r.size != n_real:
                raise ValueError(f"el representante {nombre!r} tiene {r.size} valores; se esperaban {n_real}")

        return cls(
            estrategia=d["estrategia"],
            n_bins_solicitado=int(d["n_bins_solicitado"]),
            n_bins_real=n_real,
            limites=limites,
            representantes=representantes,
            columna=d["columna"],
            n_ajuste=int(d["n_ajuste"]),
        )


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
    n_bins (entero entre 1 y 32767), columna (nombre opcional).
    Retorna: Cuantizador con limites y representantes (punto_medio, media, mediana).
    Excluye NaN y +-inf del ajuste.
    Columna constante: 1 bin con limites = [v, v].
    Cuantiles: los límites repetidos se eliminan, así que n_bins_real puede ser
    menor que n_bins_solicitado.
    Lanza ValueError si n_bins < 1 o > 32767 (los códigos son int16), si la
    estrategia no existe o si no hay valores finitos.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins debe ser un entero >= 1, se recibió {n_bins!r}")
    # NUEVO: evita el desbordamiento silencioso de los códigos int16
    if n_bins > np.iinfo(np.int16).max:
        raise ValueError(f"n_bins no puede superar {np.iinfo(np.int16).max} (los códigos son int16)")
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
    "es_nan": mascara}). Conserva la longitud de la entrada. No modifica el modelo.
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


# ----------------------------------------------------------------------------
# DataFrames (30/10)
# ----------------------------------------------------------------------------
def cuantizar_dataframe(df: pd.DataFrame, columnas: list, estrategia: str = "uniforme",
                        n_bins: int = 10):
    """Ajusta un cuantizador por columna y las sustituye por códigos int16.

    No modifica df. Retorna (df_codificado, {columna: Cuantizador}).
    Las columnas no listadas pasan intactas; el orden de columnas se conserva.
    Lanza ValueError si falta alguna columna, si hay duplicadas o si no hay columnas.
    """
    columnas = [columnas] if isinstance(columnas, str) else list(columnas)
    if not columnas:
        raise ValueError("no se indicaron columnas a cuantizar")
    if len(set(columnas)) != len(columnas):
        raise ValueError("hay columnas repetidas en la lista")
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"columnas inexistentes en el DataFrame: {faltan}")

    salida = df.copy()
    modelos = {}
    for c in columnas:
        modelo = ajustar_cuantizador(df[c], estrategia, n_bins, columna=c)
        salida[c] = transformar(df[c], modelo)
        modelos[c] = modelo
    return salida, modelos


def reconstruir_dataframe(df_codificado: pd.DataFrame, modelos: dict,
                          representante: str = "punto_medio") -> pd.DataFrame:
    """Reconstruye con los modelos guardados, sin recalcular límites.

    Las columnas que no están en `modelos` pasan intactas. No modifica la entrada.
    Lanza ValueError si un modelo apunta a una columna que no existe.
    """
    faltan = [c for c in modelos if c not in df_codificado.columns]
    if faltan:
        raise ValueError(f"columnas con modelo pero ausentes en el DataFrame: {faltan}")

    salida = df_codificado.copy()
    for c, modelo in modelos.items():
        salida[c] = reconstruir(df_codificado[c].to_numpy(), modelo, representante)
    return salida


def comparar_estadisticas(original: pd.DataFrame, reconstruido: pd.DataFrame,
                          columnas: list | None = None,
                          cuantiles=CUANTILES_ESTADISTICAS) -> dict:
    """Diferencias (reconstruido - original) en media, desviación, cuantiles y correlación.

    Retorna un dict con:
      "resumen": DataFrame con una fila por columna (dif_media, dif_std, dif_q05, ...).
      "dif_correlacion": DataFrame, corr(reconstruido) - corr(original).
      "dif_correlacion_max": máxima diferencia absoluta fuera de la diagonal.
    Si `columnas` es None se usan las numéricas comunes a ambos DataFrames.
    """
    if columnas is None:
        columnas = [c for c in original.columns
                    if c in reconstruido.columns
                    and pd.api.types.is_numeric_dtype(original[c])
                    and pd.api.types.is_numeric_dtype(reconstruido[c])]
    columnas = list(columnas)
    if not columnas:
        raise ValueError("no hay columnas numéricas comunes para comparar")
    if len(original) != len(reconstruido):
        raise ValueError("los DataFrames tienen distinto número de filas")

    o = original[columnas].astype("float64").reset_index(drop=True)
    r = reconstruido[columnas].astype("float64").reset_index(drop=True)
    qs = list(cuantiles)
    q_o, q_r = o.quantile(qs), r.quantile(qs)

    filas = []
    for c in columnas:
        fila = {"columna": c,
                "dif_media": float(r[c].mean() - o[c].mean()),
                "dif_std": float(r[c].std() - o[c].std())}
        for q in qs:
            fila[f"dif_q{int(round(q * 100)):02d}"] = float(q_r.loc[q, c] - q_o.loc[q, c])
        filas.append(fila)

    dif_corr = r.corr() - o.corr()
    if len(columnas) > 1:
        fuera_diagonal = ~np.eye(len(columnas), dtype=bool)
        dif_max = float(np.nanmax(np.abs(dif_corr.to_numpy()[fuera_diagonal])))
    else:
        dif_max = 0.0

    return {"resumen": pd.DataFrame(filas), "dif_correlacion": dif_corr,
            "dif_correlacion_max": dif_max}


def tabla_resultados(df: pd.DataFrame, columnas: list, estrategias=ESTRATEGIAS,
                     bins=(5, 10, 25, 50), representantes=REPRESENTANTES) -> pd.DataFrame:
    """Una fila por columna x estrategia x representante x bins solicitados.

    Incluye bins reales, MAE, RMSE, error máximo, diferencias estadísticas
    (media, desviación, cuantiles), diferencia máxima de correlación (global
    para esa combinación) y tiempos de ajuste, transformación y reconstrucción.
    """
    columnas = [columnas] if isinstance(columnas, str) else list(columnas)
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"columnas inexistentes en el DataFrame: {faltan}")

    original = df[columnas].astype("float64").reset_index(drop=True)
    filas = []
    for estrategia in estrategias:
        for k in bins:
            ajustes = {}
            for c in columnas:
                t0 = perf_counter()
                modelo = ajustar_cuantizador(original[c], estrategia, k, columna=c)
                t1 = perf_counter()
                codigos = transformar(original[c], modelo)
                t2 = perf_counter()
                ajustes[c] = (modelo, codigos, t1 - t0, t2 - t1)

            for rep in representantes:
                recs, t_rec = {}, {}
                for c in columnas:
                    t0 = perf_counter()
                    recs[c] = reconstruir(ajustes[c][1], ajustes[c][0], rep)
                    t_rec[c] = perf_counter() - t0

                est = comparar_estadisticas(original, pd.DataFrame(recs))
                resumen = est["resumen"].set_index("columna")

                for c in columnas:
                    modelo, _, t_aj, t_tr = ajustes[c]
                    e = calcular_error(original[c], recs[c])
                    filas.append({
                        "columna": c, "estrategia": estrategia, "representante": rep,
                        "bins_solicitados": k, "bins_reales": modelo.n_bins_real,
                        "mae": e["mae"], "rmse": e["rmse"], "error_max": e["error_max"],
                        **resumen.loc[c].to_dict(),
                        "dif_corr_max": est["dif_correlacion_max"],
                        "t_ajuste_s": t_aj, "t_transformar_s": t_tr,
                        "t_reconstruir_s": t_rec[c],
                    })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Persistencia (4/11)
# ----------------------------------------------------------------------------
def guardar_modelo(modelo: Cuantizador, ruta) -> None:
    """Guarda en JSON: version, columna, estrategia, n_bins_solicitado,
    n_bins_real, limites, representantes, n_ajuste y fecha.

    Crea las carpetas que falten. NaN se guarda como null (JSON estricto).
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    d = modelo.to_dict()
    d["fecha"] = datetime.now().isoformat(timespec="seconds")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False)


def cargar_modelo(ruta) -> Cuantizador:
    """Carga un modelo desde JSON sin recalcular nada.

    Lanza ValueError si la versión no coincide o el contenido es inconsistente.
    """
    with open(Path(ruta), "r", encoding="utf-8") as f:
        d = json.load(f)
    return Cuantizador.from_dict(d)


def guardar_modelos(modelos: dict, ruta) -> None:
    """Guarda todos los modelos de un DataFrame ({columna: Cuantizador}) en un solo JSON."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = {
        "version": VERSION_MODELO,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelos": {c: m.to_dict() for c, m in modelos.items()},
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2, allow_nan=False)


def cargar_modelos(ruta) -> dict:
    """Inverso de guardar_modelos: devuelve {columna: Cuantizador}."""
    with open(Path(ruta), "r", encoding="utf-8") as f:
        contenido = json.load(f)
    if "modelos" not in contenido:
        raise ValueError("el archivo no contiene la clave 'modelos'")
    return {c: Cuantizador.from_dict(d) for c, d in contenido["modelos"].items()}


__all__ = [
    "Cuantizador", "ESTRATEGIAS", "REPRESENTANTES",
    "ajustar_cuantizador", "transformar", "reconstruir", "calcular_error",
    "cuantizar_dataframe", "reconstruir_dataframe",
    "comparar_estadisticas", "tabla_resultados",
    "guardar_modelo", "cargar_modelo", "guardar_modelos", "cargar_modelos",
]