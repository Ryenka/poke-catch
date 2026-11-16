"""Descarga (o vuelve a descargar) los CSV de PokeAPI a datos/raw.

Uso:  python src/descargar_datos.py
"""
import pandas as pd

from pokecatch import DIR_RAW, TABLAS, URL_BASE

if __name__ == "__main__":
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    for tabla in TABLAS:
        df = pd.read_csv(URL_BASE + f"{tabla}.csv")
        df.to_csv(DIR_RAW / f"{tabla}.csv", index=False, encoding="utf-8")
        print(f"{tabla}.csv: {len(df)} filas")
