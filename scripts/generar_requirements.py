"""Escribe requirements.txt con las dependencias directas y sus versiones instaladas."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DIRECTAS = ["numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib",
            "pytest", "psutil", "jupyter", "ipykernel"]

lineas, faltan = [], []
for paquete in DIRECTAS:
    try:
        lineas.append(f"{paquete}=={version(paquete)}")
    except PackageNotFoundError:
        faltan.append(paquete)

destino = Path(__file__).resolve().parents[1] / "requirements.txt"
destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
print("\n".join(lineas))
if faltan:
    print("\nNO INSTALADOS (instálalos y vuelve a ejecutar):", faltan)