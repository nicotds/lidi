import pandas as pd

COLUMNAS = ["edad", "ingresos", "altura"]
RUTAS = {"1M": "data/dataset_1M.parquet", "5M": "data/dataset_5M.parquet"}

filas = []
for nombre, ruta in RUTAS.items():
       df = pd.read_parquet(ruta, columns=COLUMNAS)
       for col in COLUMNAS:
           s = df[col]
           filas.append({
               "dataset": nombre, "columna": col, "dtype": str(s.dtype),
               "n_filas": len(s), "pct_nan": round(100 * s.isna().mean(), 3),
               "minimo": s.min(), "maximo": s.max(), "media": s.mean(),
               "pct_ceros": round(100 * (s == 0).mean(), 3),
           })

tabla = pd.DataFrame(filas)
tabla.to_csv("resultados/actividad3/tablas/perfil_datasets.csv", index=False)
print(tabla.to_string())