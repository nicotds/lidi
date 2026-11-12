# Actividad 2

## Objetivo

Investigar y evaluar herramientas de software orientadas a la reducción y simplificación de información en conjuntos de datos tabulares de gran volumen.

## Herramientas

* Datashader
* Vaex
* DuckDB
* Polars
* Scikit-Learn

## Dataset

Se utiliza un dataset sintético de 5 millones de registros, compuesto por variables numéricas continuas, enteras y categóricas.

## Técnicas

* Agregación
* Binning
* Cuantiles
* Discretización
* Reducción de precisión numérica

## Métricas

* Tiempo de procesamiento
* Consumo de memoria
* Tamaño de los datos
* Porcentaje de reducción
* Conservación de estadísticas
* Conservación de la distribución

\## Actividad 3: cuantización de datos tabulares



Módulo propio `src/cuantizacion.py` que discretiza columnas numéricas en bins, reconstruye

valores aproximados y mide el error. Scikit-Learn (`KBinsDiscretizer`) se usa solo como

referencia de validación.



\### Estructura



```text

src/cuantizacion.py          módulo (código propio)

tests/                       pruebas (pytest)

scripts/ejecutar\_dataset.py  corrida completa sobre un Parquet

scripts/validacion\_sklearn.py   comparación con KBinsDiscretizer

scripts/verificar\_reproducibilidad.py   recarga de JSON y comparación de códigos

notebooks/actividad3/        notebooks (demo\_cuantizacion.ipynb es la demostración)

resultados/actividad3/       modelos/, tablas/, graficos/ (datos\_cuantizados/ no se versiona)

```



\### Instalación y pruebas



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python -m pytest -q          # <N> passed

```



Entorno usado: Python 3.12, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, PyArrow 20.0.0.

`requirements\_actividad2.txt` conserva las dependencias de la Actividad 2.



\### Uso



```python

from src.cuantizacion import (ajustar\_cuantizador, transformar, reconstruir, calcular\_error,

&#x20;                             guardar\_modelo, cargar\_modelo)



m = ajustar\_cuantizador(x, "cuantiles", 10, columna="ingresos")  # "uniforme" o "cuantiles"

codigos = transformar(x, m)                                     # int16; NaN -> -1

aprox = reconstruir(codigos, m, "mediana")                      # punto\_medio | media | mediana

print(calcular\_error(x, aprox))                                 # mae, rmse, error\_max, n\_validos

guardar\_modelo(m, "modelo.json"); m = cargar\_modelo("modelo.json")

```



Para DataFrames: `cuantizar\_dataframe`, `reconstruir\_dataframe`, `comparar\_estadisticas`,

`tabla\_resultados`. Para varias columnas: `guardar\_modelos` / `cargar\_modelos`.



\### Convenciones



\- Intervalos `\[e\_i, e\_{i+1})`; el último incluye el máximo.

\- NaN -> código `-1`; al reconstruir vuelve a NaN.

\- Fuera de rango: menor al mínimo -> bin 0, mayor al máximo -> último bin; se devuelve la máscara `fuera\_rango`.

\- Columna constante: un solo bin. Cuantiles con límites repetidos: se eliminan, así que `n\_bins\_real` puede ser menor al solicitado.

\- Cada modelo guarda los tres representantes (punto medio, media, mediana). Un bin vacío usa el punto medio.

\- `±inf` se excluye del ajuste. Los códigos son `int16`, por lo que `n\_bins <= 32767`.

\- JSON con campo `version`; cargar un modelo no recalcula nada.



\### Validación frente a scikit-learn 1.9.1



Con `subsample=None` y `quantile\_method="linear"`, los límites y los códigos coinciden con

`KBinsDiscretizer` en todos los casos probados (columnas reales, 70 % de ceros, con NaN):

diferencia máxima de límites \~1e-11 y 100 % de códigos iguales. Con la columna constante los

bordes se representan distinto (sklearn usa `\[-inf, inf]`, el módulo `\[v, v]`), pero los

códigos son iguales. Con los parámetros por defecto de scikit-learn (`subsample=200000`,

`quantile\_method='averaged\_inverted\_cdf'`) los códigos difieren entre 0.8 % y 1.7 % en la

estrategia de cuantiles con 1M de filas. Detalle en `resultados/actividad3/tablas/validacion\_sklearn.csv`.



\### Resultados (1M y 5M de filas; bins 5, 10, 25, 50)



Fuente: `resultados/actividad3/tablas/resultados\_actividad3.csv` (144 filas).



<PEGAR AQUÍ el contenido de resultados/actividad3/tablas/resumen\_readme.md>



Observaciones: <ESCRIBE 3 o 4 con lo que viste: cómo baja el MAE al subir los bins, qué

estrategia gana en `ingresos`, cuánto escalan los tiempos de 1M a 5M, RAM máxima>.



\### Limitaciones



\- `edad` tiene valores negativos en los datos sintéticos (mínimo cercano a -18); se cuantiza igual, sin corregirlos.

\- `ingresos` es asimétrica: con `uniforme` casi todos los datos caen en los primeros bins y el máximo del 5M, mayor que el del 1M, ensancha los bins. `cuantiles` equilibra los bins pero deja un último bin ancho.

\- Los límites uniformes usan `np.linspace`; un valor justo en un borde puede moverse al bin vecino por redondeo.

\- La mediana por bin ordena la columna completa: usa memoria adicional proporcional al número de filas.

\- `experiencia` y `categoria` no se cuantizan en esta versión.

\- No se implementó k-means 1D (Lloyd-Max); queda como mejora para una versión posterior.



\### Reproducir las corridas



```powershell

python scripts\\ejecutar\_dataset.py --ruta data\\dataset\_1M.parquet --etiqueta 1M

python scripts\\ejecutar\_dataset.py --ruta data\\dataset\_5M.parquet --etiqueta 5M

python scripts\\verificar\_reproducibilidad.py

python scripts\\validacion\_sklearn.py

```



Los Parquet de `data/` y de `datos\_cuantizados/` no se suben al repositorio; se regeneran con los comandos anteriores.

