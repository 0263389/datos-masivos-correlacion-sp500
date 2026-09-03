"""Configuracion central del proyecto.

Este modulo resuelve dos problemas:

1. **Rutas portables.** Todas las rutas se derivan de la ubicacion de este
   archivo en disco, no de la carpeta desde la que se ejecuta el codigo ni de
   la computadora de un integrante del equipo. El repositorio se puede clonar
   en cualquier lado y todo sigue funcionando.

2. **Un solo lugar para los parametros.** Fechas, universo y URLs viven aqui,
   no regados por el notebook. Cambiar el periodo de analisis es cambiar una
   linea.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
# __file__ es la ruta de ESTE archivo (src/config.py).
#   .resolve()  -> la vuelve absoluta y resuelve enlaces simbolicos
#   .parents[0] -> src/
#   .parents[1] -> la raiz del repositorio
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # descargas crudas (ignoradas por git)
PROCESSED_DIR = DATA_DIR / "processed"  # respaldo pequeno (si se versiona)
FIGURES_DIR = ROOT / "figures"       # graficas exportadas para el README

for _directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Carga las variables de entorno desde .env si el archivo existe.
# Si no existe, no falla: las fuentes que no requieren llave siguen sirviendo.
load_dotenv(ROOT / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# --------------------------------------------------------------------------
# Parametros del analisis
# --------------------------------------------------------------------------
# Periodo. Arranca en 2015 para tener ~10 anios: suficiente para estimar
# correlaciones estables e incluye el desplome de COVID (marzo 2020), que
# usaremos en entregables posteriores para estudiar correlaciones en crisis.
START_DATE = "2015-01-01"
END_DATE = None  # None = hasta el dia mas reciente disponible

# Ticker del indice, usado como referencia del "mercado".
MARKET_TICKER = "^GSPC"  # indice S&P 500

# Fuente de la lista de constituyentes y su clasificacion sectorial GICS.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Minimo de observaciones para que un ticker entre al analisis.
#
# Este numero es mas delicado de lo que parece. El analisis usa un "panel
# balanceado" (una tabla rectangular sin huecos), y en un panel balanceado
# **la longitud la impone el ticker sobreviviente mas joven**: si dejamos
# entrar una empresa que salio a bolsa en 2024, todas las demas pierden la
# historia anterior a 2024.
#
# Se midio el intercambio empresas/historia sobre los datos reales:
#
#     minimo   empresas   dias del panel   % de historia
#        500        499              609           20.8%
#       2000        475             2030           69.2%
#       2900        459             2932           99.9%
#       2934        457             2934          100.0%
#
# Exigir historia casi completa cuesta 44 empresas (9% del universo) y a cambio
# multiplica por cinco la historia disponible. Se elige 2900 en lugar de 2934
# para tolerar que alguna accion haya tenido uno o dos dias suspendidos.
MIN_OBSERVATIONS = 2900  # ~99% del periodo 2015-2026

# Umbral de rendimiento diario a partir del cual el dato es sospechoso.
EXTREME_RETURN_THRESHOLD = 0.50

# Numero de eventos extremos que tolera un ticker antes de considerarse corrupto.
#
# Criterio empirico observado en los datos: una empresa real tiene a lo mucho un
# evento de +/-50% en once anios (una bancarrota, un desplome sectorial, un
# reporte de vendedores en corto). Un ticker con muchos de esos, y ademas
# reversibles, tiene un split mal aplicado por la fuente. Ejemplo real
# encontrado: MNST (Monster Beverage) oscila entre ~95 y ~47 dolares -
# exactamente la mitad - en dias sueltos de julio y agosto de 2026.
MAX_EXTREME_EVENTS = 2

# --------------------------------------------------------------------------
# Archivos de cache
# --------------------------------------------------------------------------
# Cache de trabajo. Vive en data/raw/, que esta en .gitignore: son archivos
# que se pueden regenerar y no tiene sentido versionarlos.
PRICES_CACHE = RAW_DIR / "sp500_prices.parquet"
CONSTITUENTS_CACHE = RAW_DIR / "sp500_constituents.parquet"
MACRO_CACHE = RAW_DIR / "macro_fred.parquet"

# Respaldo que SI viaja con el repositorio (data/processed/ no esta ignorado).
#
# Existe por una razon concreta: quien clone el repositorio recibe data/raw/
# vacio, asi que el codigo tiene que descargar todo en vivo. Eso funciona, pero
# deja la ejecucion a merced de que Yahoo Finance este disponible ese dia. El
# respaldo garantiza que el notebook siempre se pueda ejecutar.
#
# Se guarda en float32 y con compresion zstd para que ocupe lo menos posible:
# la rubrica pide subir solo datos pequenios o muestras.
PRICES_FALLBACK = PROCESSED_DIR / "sp500_prices_fallback.parquet"
CONSTITUENTS_FALLBACK = PROCESSED_DIR / "sp500_constituents.parquet"
MACRO_FALLBACK = PROCESSED_DIR / "macro_fred.parquet"
