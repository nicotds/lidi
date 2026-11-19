# Especificación del benchmark de cuantización (Actividad 4)

## Objetivo
Evaluar con las mismas reglas cada idea de cuantización de datos tabulares, reutilizando lo medido
en la Actividad 3 (módulo propio `src/cuantizacion.py`) y añadiendo distribuciones controladas,
métricas de distribución, repeticiones y generalización.

## Preguntas

| # | Pregunta | Cómo se responde |
|---|---|---|
| P1 | ¿Cuánto baja el error al subir los bins y dónde se estanca? | NRMSE contra bins (escala log) por método y distribución |
| P2 | ¿Qué estrategia gana según la forma de la distribución? | Ranking medio por distribución y bins |
| P3 | ¿Media o mediana mejoran al punto medio, y cuánto? | Ganancia relativa en NRMSE y NMAE sobre el punto medio |
| P4 | ¿Cómo se compara la implementación propia con scikit-learn? | Diferencia de error y aceleración de tiempo |
| P5 | ¿Qué conviene por unidad de compresión o de tiempo? | Frente de Pareto error-bits y error-tiempo |
| P6 | ¿Los límites ajustados generalizan y resisten outliers? | Cambio de error ajuste->prueba, % fuera de rango, caso con outliers |

## Métodos (9 + 2 controles)

| Grupo | Método | Detalle |
|---|---|---|
| Propio | uniforme + punto medio / media / mediana | mismo ajuste; solo cambia la reconstrucción |
| Propio | cuantiles + punto medio / media / mediana | mismo ajuste; solo cambia la reconstrucción |
| Referencia | `KBinsDiscretizer` uniform, quantile, kmeans | `subsample=None`, `quantile_method="linear"`, `random_state=42` |
| Control | float32 | sin bins |
| Control | sin cuantizar | error 0; da tiempo y memoria base |

Notas: (1) `inverse_transform` de scikit-learn reconstruye con el punto medio de los límites,
también en kmeans (no usa los centroides). (2) El ajuste del módulo propio calcula los tres
representantes, por eso su tiempo de ajuste es igual en las tres variantes. (3) Con columna
constante scikit-learn usa bordes [-inf, inf] y su `inverse_transform` da NaN; el adaptador
reconstruye con el valor constante.

## Datasets
- Reales: `dataset_1M.parquet` y `dataset_5M.parquet`; columnas `edad`, `ingresos`, `altura`.
- Sintéticos (1 columna `valor`, 1M filas, semilla 42, 2 % de NaN): normal N(50,10); lognormal
  (mu=10, sigma=0,6); uniforme [0,100]; bimodal (50/50 de N(30,5) y N(70,5)); normal con outliers
  (0,1 % de valores a 50 desviaciones); ceros (70 % de ceros y resto normal positivo; tras los NaN
  quedan ~68,6 % de ceros).
- Bins: 5, 10, 25, 50, 100 y 256.
- Particiones: ajuste con el 50 % de las filas y prueba con el otro 50 % (semilla 42). La partición
  "ajuste" se evalúa sobre las filas de ajuste; "prueba", sobre las otras.
- Casos: 9 métodos x 6 bins + 2 controles = 56 por columna; 12 columnas (6 sintéticas + 3 + 3 reales).

## Métricas
Todas sobre pares finitos (sin NaN). Desviación poblacional (ddof=0).

| Familia | Métrica | Cálculo |
|---|---|---|
| Error puntual | NMAE, NRMSE, error máximo normalizado | MAE / rango, RMSE / desviación, máximo / rango |
| Distribución | Wasserstein-1 normalizado, KS, distorsión de histograma | W1 / desviación; estadístico de `ks_2samp`; L1 entre histogramas normalizados de 50 bins sobre el rango original |
| Estructura | Error de correlación | Máximo de la diferencia absoluta entre matrices de Pearson (3 columnas reales) |
| Eficiencia | Tiempos de ajuste, codificación, decodificación; memoria pico | Mediana de 3 repeticiones tras 1 de calentamiento; `tracemalloc` |
| Compresión | Bits por valor y razón | bits = ceil(log2(bins_reales + 1)); razón = 64 / bits (float32: 32 bits; sin cuantizar: 64) |
| Generalización | Cambio de error ajuste->prueba, % fuera de rango | % de valores de la partición fuera de [límite inferior, límite superior] |

Normalizadas = NaN (sin dividir por cero) si el rango o la desviación es 0.
En el 5M las métricas de distribución se calculan sobre una muestra fija de 200 000 filas por partición.

## Protocolo de medición
`time.perf_counter`; `gc.collect()` antes de cada medición; 1 calentamiento y 3 repeticiones
(mediana); datos cargados fuera del cronómetro; `OMP_NUM_THREADS=1`. Los tiempos se miden sobre la
mitad de ajuste y se repiten en las dos filas del caso. La memoria es el pico de `tracemalloc`
durante ajustar + codificar + decodificar: mide asignaciones de NumPy/Python, no la memoria total.

## Reproducibilidad
Semilla 42 en todo. `metadata.json` con versiones (Python, NumPy, pandas, SciPy, scikit-learn),
CPU y RAM. La configuración (`config.yaml`) se congela el 19/11 con la etiqueta `benchmark-v1`.

## Esquema de salida (tabla larga: una fila por caso y partición)
`dataset, columna, metodo, estrategia, representante, bins_solicitados, bins_reales, particion,
n_filas, nmae, nrmse, err_max_norm, w1_norm, ks, dist_hist, err_corr, t_ajuste_s, t_codificar_s,
t_decodificar_s, mem_pico_mb, bits_por_valor, razon_compresion, pct_fuera_rango, t_caso_s`.
`err_corr` solo existe en los datasets de varias columnas (reales). En los controles los bins son NaN.

## Agregación
Ranking por caso (1 = menor NRMSE) promediado por método, y frente de Pareto. Sin puntaje compuesto.

## Limitaciones conocidas
- Memoria medida solo con `tracemalloc`. Métricas de distribución sobre muestra en el 5M.
- Datos sintéticos. `dist_hist` es exactamente 0 para uniforme con 50 bins porque su rejilla
  coincide con la del histograma (artefacto de la métrica).

## Decisión sobre kmeans (19/11)
kmeans_max_ajuste = 200000: kmeans se ajusta con una muestra fija de 200 000 filas (semilla 42)
porque la estimación del piloto lo hacía inviable en el 5M. La codificación y la decodificación
usan todas las filas. Esto favorece el tiempo de ajuste de kmeans y puede subir un poco su error.
