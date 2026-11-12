#### MAE con representante `media`

| dataset | columna | estrategia | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| 1M | altura | cuantiles | 2.3 | 1.275 | 0.5711 | 0.3065 |
| 1M | altura | uniforme | 4.199 | 2.375 | 0.9928 | 0.4988 |
| 1M | edad | cuantiles | 2.762 | 1.533 | 0.6854 | 0.3681 |
| 1M | edad | uniforme | 4.844 | 2.719 | 1.133 | 0.5696 |
| 1M | ingresos | cuantiles | 3148 | 1828 | 872.4 | 491.7 |
| 1M | ingresos | uniforme | 8072 | 4703 | 2274 | 1196 |
| 5M | altura | cuantiles | 2.302 | 1.277 | 0.5715 | 0.3071 |
| 5M | altura | uniforme | 4.256 | 2.413 | 1.011 | 0.5086 |
| 5M | edad | cuantiles | 2.762 | 1.531 | 0.6855 | 0.3682 |
| 5M | edad | uniforme | 5.04 | 2.85 | 1.191 | 0.5992 |
| 5M | ingresos | cuantiles | 3139 | 1824 | 868 | 488.5 |
| 5M | ingresos | uniforme | 8831 | 5505 | 2692 | 1449 |

#### Error máximo con representante `media`

| dataset | columna | estrategia | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| 1M | altura | cuantiles | 36.34 | 32.79 | 28.81 | 26.13 |
| 1M | altura | uniforme | 17.19 | 7.329 | 2.731 | 1.187 |
| 1M | edad | cuantiles | 41.14 | 36.87 | 32.06 | 28.85 |
| 1M | edad | uniforme | 19.47 | 8.825 | 3.023 | 1.379 |
| 1M | ingresos | cuantiles | 2.004e+05 | 1.919e+05 | 1.802e+05 | 1.71e+05 |
| 1M | ingresos | uniforme | 3.742e+04 | 1.61e+04 | 7386 | 3828 |
| 5M | altura | cuantiles | 37.19 | 33.63 | 29.65 | 26.98 |
| 5M | altura | uniforme | 17.62 | 8.083 | 3.019 | 1.213 |
| 5M | edad | cuantiles | 43.61 | 39.35 | 34.57 | 31.37 |
| 5M | edad | uniforme | 20.66 | 9.334 | 3.125 | 1.437 |
| 5M | ingresos | cuantiles | 2.538e+05 | 2.453e+05 | 2.337e+05 | 2.245e+05 |
| 5M | ingresos | uniforme | 4.71e+04 | 2.057e+04 | 9001 | 4791 |

#### Tiempos (s, suma de las 3 columnas), ajuste y transformación

| dataset | estrategia | bins_solicitados | t_ajuste_s | t_transformar_s |
|---|---|---|---|---|
| 1M | cuantiles | 5 | 0.1683 | 0.05146 |
| 1M | cuantiles | 10 | 0.1896 | 0.07423 |
| 1M | cuantiles | 25 | 0.2422 | 0.1114 |
| 1M | cuantiles | 50 | 0.3201 | 0.1177 |
| 1M | uniforme | 5 | 0.2069 | 0.05238 |
| 1M | uniforme | 10 | 0.1245 | 0.06005 |
| 1M | uniforme | 25 | 0.1793 | 0.1001 |
| 1M | uniforme | 50 | 0.1666 | 0.1019 |
| 5M | cuantiles | 5 | 0.9791 | 0.2748 |
| 5M | cuantiles | 10 | 1.185 | 0.4172 |
| 5M | cuantiles | 25 | 1.649 | 0.7302 |
| 5M | cuantiles | 50 | 2.174 | 0.9578 |
| 5M | uniforme | 5 | 0.6952 | 0.1901 |
| 5M | uniforme | 10 | 0.7346 | 0.3638 |
| 5M | uniforme | 25 | 0.8212 | 0.4981 |
| 5M | uniforme | 50 | 0.9158 | 0.5655 |

#### Memoria y tamaño de códigos

| dataset | rss_max_MB | codigos_MB_max |
|---|---|---|
| 1M | 232.7 | 2.259 |
| 5M | 673.4 | 11.29 |

Bins reales distintos de los solicitados: 0 filas
