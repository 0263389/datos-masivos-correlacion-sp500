# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 — Bronze: ingesta al data lake
# MAGIC
# MAGIC **Entregable 2 · Capa Bronze · Responsable: Franco Rissotto**
# MAGIC
# MAGIC Trae los datos crudos de las tres fuentes y los deja como tablas Delta en
# MAGIC Unity Catalog. **Aquí no se limpia nada**: Bronze guarda lo que la fuente
# MAGIC devolvió, tal cual, incluidos sus errores.
# MAGIC
# MAGIC Esa es la regla de la arquitectura medallion y no es un capricho: si más
# MAGIC adelante descubrimos que una regla de limpieza estaba mal, podemos
# MAGIC recalcular Silver sin volver a descargar nada. Si limpiáramos aquí, el dato
# MAGIC original se perdería para siempre.
# MAGIC
# MAGIC | Tabla | Qué contiene | Fuente |
# MAGIC |---|---|---|
# MAGIC | `bronze.constituents` | 1,503 empresas del S&P 1500 con sector y tamaño | Wikipedia (scraping) |
# MAGIC | `bronze.prices` | Precios diarios ajustados, formato largo | Yahoo Finance |
# MAGIC | `bronze.macro` | VIX y tasa del Tesoro a 10 años | FRED (API) |

# COMMAND ----------

# MAGIC %pip install -q yfinance

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType, FloatType, StringType, StructField, StructType,
)

CATALOGO = "workspace"
ESQUEMA = "bronze"
INICIO = "2015-01-01"

# Wikipedia rechaza las peticiones sin User-Agent de navegador.
UA = {"User-Agent": "Mozilla/5.0 (compatible; proyecto-academico/1.0)"}

INDICES = {
    "large": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "mid": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "small": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{ESQUEMA}")
spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {ESQUEMA}")

print(f"Escribiendo en {CATALOGO}.{ESQUEMA}")
print(f"Spark {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Constituyentes del S&P Composite 1500
# MAGIC
# MAGIC Tres páginas de Wikipedia, una por índice de capitalización. Comparten las
# MAGIC mismas columnas, así que el mismo scraping sirve para las tres.
# MAGIC
# MAGIC La columna `cap_tier` (large / mid / small) es la que habilita la pregunta
# MAGIC de este entregable: **¿dependen menos del mercado las empresas chicas?**

# COMMAND ----------

# ──────────────────────────────────────────────────────────────────────
# Nota sobre la duplicacion con src/data.py
#
# Las funciones de scraping de abajo repiten lo que ya hace
# `src.data.get_sp1500_constituents()`. La duplicacion es deliberada.
#
# Este notebook es el punto de entrada del data lake: debe poder construir la
# capa Bronze desde cero aunque se ejecute suelto, sin que el repositorio este
# montado como Git folder. Importar desde src/ lo volveria dependiente de una
# ruta que en una ejecucion programada puede no existir.
#
# Los notebooks 05 y 06 si importan de src/ -- pero ellos leen tablas que
# Bronze ya escribio, asi que fallar ahi es recuperable. Fallar en la ingesta
# deja el lake vacio.
#
# El costo: si Wikipedia cambia la estructura de sus tablas, hay que
# arreglarlo en los dos lados.
# ──────────────────────────────────────────────────────────────────────
def a_ticker_yahoo(simbolo: str) -> str:
    """Wikipedia escribe BRK.B; Yahoo escribe BRK-B.

    Sin esta traducción esos tickers regresan vacíos de Yahoo **sin lanzar
    ningún error** y desaparecen del análisis en silencio.
    """
    return simbolo.strip().upper().replace(".", "-")


def descargar_indice(url: str, cap_tier: str) -> pd.DataFrame:
    respuesta = requests.get(url, headers=UA, timeout=30)
    respuesta.raise_for_status()

    # Se busca la tabla por sus columnas en lugar de asumir que es la primera:
    # si Wikipedia reordena la página, esto sigue funcionando.
    tabla = None
    for candidata in pd.read_html(StringIO(respuesta.text)):
        columnas = [str(c) for c in candidata.columns]
        if any(c in ("Symbol", "Ticker") for c in columnas) and "GICS Sector" in columnas:
            tabla = candidata
            break
    if tabla is None:
        raise RuntimeError(f"No se encontró la tabla de constituyentes en {url}")

    col_simbolo = "Symbol" if "Symbol" in tabla.columns else "Ticker"
    return pd.DataFrame({
        "ticker": tabla[col_simbolo].map(a_ticker_yahoo),
        "company": tabla["Security"].astype(str),
        "sector": tabla["GICS Sector"].astype(str),
        "sub_industry": tabla["GICS Sub-Industry"].astype(str),
        "cap_tier": cap_tier,
    })


partes = [descargar_indice(url, tier) for tier, url in INDICES.items()]
constituyentes = pd.concat(partes, ignore_index=True)

# Un ticker en dos índices a la vez se queda con el de mayor capitalización:
# duplicarlo rompería el conteo de pares únicos más adelante.
orden = {"large": 0, "mid": 1, "small": 2}
constituyentes["_orden"] = constituyentes["cap_tier"].map(orden)
constituyentes = (
    constituyentes.sort_values(["ticker", "_orden"])
    .drop_duplicates(subset="ticker", keep="first")
    .drop(columns="_orden")
    .sort_values("ticker")
    .reset_index(drop=True)
)

print(f"{len(constituyentes)} tickers únicos")
print(constituyentes["cap_tier"].value_counts().to_string())

# COMMAND ----------

(
    spark.createDataFrame(constituyentes)
    .withColumn("ingesta_ts", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOGO}.{ESQUEMA}.constituents")
)

display(spark.table(f"{CATALOGO}.{ESQUEMA}.constituents").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Precios diarios ajustados
# MAGIC
# MAGIC `auto_adjust=True` es obligatorio: devuelve precios corregidos por splits y
# MAGIC dividendos. Con el cierre crudo, un split 4:1 se vería como una caída de
# MAGIC −75% en un día y contaminaría toda la matriz de correlación.
# MAGIC
# MAGIC **La descarga va por lotes**, no de golpe. Tres razones: el driver no tiene
# MAGIC que sostener 1,500 tickers en memoria, un fallo de red no tira las dos horas
# MAGIC de trabajo, y se puede reanudar donde se quedó.
# MAGIC
# MAGIC **El formato es largo** (`fecha, ticker, precio`), no ancho. Es el formato
# MAGIC nativo de Spark: permite particionar por año y filtrar sin leer todo.

# COMMAND ----------

ESQUEMA_PRECIOS = StructType([
    StructField("date", DateType(), False),
    StructField("ticker", StringType(), False),
    StructField("close", FloatType(), True),
])

TAMANIO_LOTE = 150


def a_formato_largo(ancho: pd.DataFrame) -> pd.DataFrame:
    """Convierte la tabla ancha de yfinance a filas (fecha, ticker, precio)."""
    if isinstance(ancho.columns, pd.MultiIndex):
        for nivel in range(ancho.columns.nlevels):
            if "Close" in ancho.columns.get_level_values(nivel):
                ancho = ancho.xs("Close", axis=1, level=nivel)
                break
    largo = ancho.stack().dropna().rename("close").reset_index()
    largo.columns = ["date", "ticker", "close"]
    largo["date"] = pd.to_datetime(largo["date"]).dt.date
    largo["close"] = largo["close"].astype("float32")
    return largo


tickers = constituyentes["ticker"].tolist()
lotes = [tickers[i:i + TAMANIO_LOTE] for i in range(0, len(tickers), TAMANIO_LOTE)]

primer_lote = True
filas_totales = 0
fallidos = []

for n, lote in enumerate(lotes, start=1):
    try:
        crudo = yf.download(
            tickers=lote, start=INICIO, auto_adjust=True,
            progress=False, threads=True,
        )
        if crudo.empty:
            fallidos.extend(lote)
            print(f"  lote {n}/{len(lotes)}: vacío")
            continue

        largo = a_formato_largo(crudo)
        sdf = spark.createDataFrame(largo, schema=ESQUEMA_PRECIOS)

        (
            sdf.write.mode("overwrite" if primer_lote else "append")
            .format("delta")
            .saveAsTable(f"{CATALOGO}.{ESQUEMA}.prices")
        )
        primer_lote = False
        filas_totales += len(largo)
        print(f"  lote {n}/{len(lotes)}: {len(largo):,} filas  (acumulado {filas_totales:,})")

    except Exception as error:
        fallidos.extend(lote)
        print(f"  lote {n}/{len(lotes)}: FALLÓ -> {type(error).__name__}")

print(f"\nTotal: {filas_totales:,} filas")
if fallidos:
    print(f"Tickers sin datos ({len(fallidos)}): {fallidos}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Particionado por año
# MAGIC
# MAGIC Partir la tabla por año significa que una consulta sobre 2020 lee solo los
# MAGIC archivos de 2020 en lugar de los 4 millones de filas. Se llama *partition
# MAGIC pruning* y es de lo que más impacto tiene en el rendimiento.
# MAGIC
# MAGIC El año es buena clave de partición porque tiene **12 valores** (ni muy pocos
# MAGIC ni demasiados) y casi todas las consultas filtran por fecha. Particionar por
# MAGIC `ticker` sería un error: 1,500 particiones diminutas, cada una un archivo
# MAGIC chiquito, y Spark sufre con muchos archivos pequeños.

# COMMAND ----------

(
    spark.table(f"{CATALOGO}.{ESQUEMA}.prices")
    .withColumn("year", F.year("date"))
    .withColumn("ingesta_ts", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("year")
    .format("delta")
    .saveAsTable(f"{CATALOGO}.{ESQUEMA}.prices")
)

precios = spark.table(f"{CATALOGO}.{ESQUEMA}.prices")
print(f"filas: {precios.count():,}")
print(f"tickers: {precios.select('ticker').distinct().count():,}")
display(
    precios.groupBy("year").count().orderBy("year")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Series macro de FRED
# MAGIC
# MAGIC Requiere una llave gratuita. **Nunca se escribe en el código**: se guarda
# MAGIC como un *secret* de Databricks o se pega aquí temporalmente sin hacer commit.
# MAGIC
# MAGIC Si no hay llave, esta sección avisa y continúa. El resto del pipeline no
# MAGIC depende de ella.

# COMMAND ----------

try:
    FRED_API_KEY = dbutils.secrets.get(scope="proyecto", key="fred_api_key")
except Exception:
    FRED_API_KEY = ""  # <- pega tu llave aquí si no configuraste secrets

SERIES = {"vix": "VIXCLS", "treasury_10y": "DGS10"}

if not FRED_API_KEY:
    print("Sin FRED_API_KEY. Se omite la tabla macro; el pipeline continúa.")
else:
    columnas = {}
    for nombre, serie_id in SERIES.items():
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": serie_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "observation_start": INICIO},
            timeout=30,
        )
        r.raise_for_status()
        obs = pd.DataFrame(r.json()["observations"])
        # FRED marca los faltantes con un punto ".", no con nulos.
        valores = pd.to_numeric(obs["value"], errors="coerce")
        valores.index = pd.to_datetime(obs["date"])
        columnas[nombre] = valores

    macro = pd.DataFrame(columnas).reset_index()
    macro.columns = ["date", *SERIES.keys()]
    macro["date"] = macro["date"].dt.date

    (
        spark.createDataFrame(macro)
        .withColumn("ingesta_ts", F.current_timestamp())
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOGO}.{ESQUEMA}.macro")
    )
    print(f"macro: {len(macro):,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificación de la capa Bronze

# COMMAND ----------

for tabla in ["constituents", "prices", "macro"]:
    nombre = f"{CATALOGO}.{ESQUEMA}.{tabla}"
    try:
        n = spark.table(nombre).count()
        print(f"  {tabla:15} {n:>10,} filas")
    except Exception:
        print(f"  {tabla:15} {'no existe':>10}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué sigue
# MAGIC
# MAGIC `02_silver_limpieza` toma estas tablas y produce la capa Silver: panel
# MAGIC balanceado, tickers corruptos excluidos y rendimientos logarítmicos
# MAGIC calculados con *window functions*.
