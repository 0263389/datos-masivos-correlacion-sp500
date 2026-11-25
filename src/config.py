"""Configuracion central del proyecto.

Este modulo resuelve dos problemas:

1. **Rutas portables.** Todas las rutas se derivan de la ubicacion de este
   archivo en disco, no de la carpeta desde la que se ejecuta el codigo ni de
   la computadora de un integrante del equipo. El repositorio se puede clonar
   en cualquier lado y todo sigue funcionando.

2. **Un solo lugar para los parametros.** Fechas, universo y URLs viven aqui,
   no regados por el notebook. Cambiar el periodo de analisis es cambiar una
   linea.

3. **Un modo reproducible sin red.** Con ``SP500_SOLO_RESPALDO=1`` la capa de
   datos no intenta descargar nada y usa unicamente los parquet incluidos en
   el repositorio. Es lo que permite que la bateria de pruebas de un resultado
   identico en cualquier maquina, y que la integracion continua no dependa de
   que Yahoo Finance responda.
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

# No hay un ticker del indice. El "mercado" de este proyecto es el promedio
# equiponderado de los activos del panel, que calcula `cofactors.mercado`.
#
# Es una decision, no una omision: el S&P 500 pondera por capitalizacion, y
# sin capitalizaciones actualizadas dia por dia no se puede reproducir esa
# ponderacion. Un indice descargado tampoco serviria como referencia limpia,
# porque contendria empresas que no estan en el panel. El promedio simple de
# los mismos activos que se analizan es la referencia honesta, y tiene una
# consecuencia util: la beta promedio contra el vale exactamente 1.

# Fuente de la lista de constituyentes y su clasificacion sectorial GICS.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Universo ampliado para el Entregable 2: el S&P Composite 1500, que es la
# union de los tres indices por tamanio de capitalizacion. Las tres paginas de
# Wikipedia comparten las mismas cuatro columnas que nos interesan, asi que el
# mismo scraping sirve para las tres.
#
# El salto no es solo de volumen. Permite preguntar algo que con el S&P 500
# solo no se puede: las empresas chicas dependen menos del mercado que las
# grandes? El tamanio de capitalizacion entra como variable de analisis.
SP_INDEX_URLS = {
    "large": SP500_WIKI_URL,
    "mid": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "small": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

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
# Modo reproducible: ignora la descarga en vivo y usa solo los respaldos del
# repositorio. Sin esto, una maquina sin cache -la de integracion continua, o
# la de un companiero que acaba de clonar- bajaria precios frescos de Yahoo y
# obtendria cifras distintas a las que el README publica, de modo que las
# pruebas de reproducibilidad fallarian por la razon equivocada.
SOLO_RESPALDO = os.getenv("SP500_SOLO_RESPALDO", "").strip().lower() in {
    "1", "true", "si", "sí", "yes",
}

MIN_OBSERVATIONS = 2900  # ~99% del periodo 2015-2026

# Umbral de rendimiento diario a partir del cual el dato es sospechoso.
EXTREME_RETURN_THRESHOLD = 0.50

# Deteccion de datos corruptos por reversion. Un movimiento extremo cuenta como
# error de la fuente solo si el precio REGRESA a donde estaba poco despues: un
# split aplicado de forma inconsistente va y vuelve, mientras que una quiebra o
# un desplome de verdad no se deshacen.
#
# Son los mismos valores que usa databricks/02_silver_limpieza.py, y estan aqui
# para que las dos implementaciones no puedan separarse sin que se note.
TOLERANCIA_REVERSION = 0.15   # volver a +/-15% del precio previo cuenta como regreso
DIAS_REVERSION = 3            # la reversion no siempre ocurre al dia siguiente
MAX_REVERSIONES = 1           # mas de una ya es patron, no accidente

# --------------------------------------------------------------------------
# Archivos de cache
# --------------------------------------------------------------------------
# Cache de trabajo. Vive en data/raw/, que esta en .gitignore: son archivos
# que se pueden regenerar y no tiene sentido versionarlos.
PRICES_CACHE = RAW_DIR / "sp500_prices.parquet"
CONSTITUENTS_CACHE = RAW_DIR / "sp500_constituents.parquet"
SP1500_CONSTITUENTS_CACHE = RAW_DIR / "sp1500_constituents.parquet"
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
SP1500_CONSTITUENTS_FALLBACK = PROCESSED_DIR / "sp1500_constituents.parquet"
MACRO_FALLBACK = PROCESSED_DIR / "macro_fred.parquet"
