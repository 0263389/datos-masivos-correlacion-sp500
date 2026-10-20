# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: limpieza y rendimientos
# MAGIC
# MAGIC **Entregable 2 · Capa Silver · Responsable: Miguel Herrera**
# MAGIC
# MAGIC Toma las tablas crudas de Bronze y produce datos listos para analizar. Aquí
# MAGIC sí se limpia: se auditan los datos, se descartan los tickers sin historia
# MAGIC suficiente o con precios corruptos, y se calculan los rendimientos
# MAGIC logarítmicos.
# MAGIC
# MAGIC **La pieza central son las *window functions***. Calcular un rendimiento
# MAGIC necesita comparar el precio de hoy con el de ayer, y "ayer" solo tiene
# MAGIC sentido *dentro* de cada ticker y *ordenado* por fecha. Eso es exactamente
# MAGIC lo que hace una ventana: `partitionBy("ticker").orderBy("date")`.
# MAGIC
# MAGIC Sin ventanas habría que hacer un join de la tabla consigo misma desplazada
# MAGIC un día — mucho más caro y mucho más fácil de equivocar.
# MAGIC
# MAGIC | Tabla que produce | Qué contiene |
# MAGIC |---|---|
# MAGIC | `silver.quality_report` | Diagnóstico por ticker: historia, nulos, anomalías |
# MAGIC | `silver.returns` | Rendimientos logarítmicos del panel limpio |

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOGO = "workspace"
BRONZE = f"{CATALOGO}.bronze"
SILVER = f"{CATALOGO}.silver"

# Un movimiento diario de +/-50% en una empresa de indice casi siempre es un
# split mal aplicado por la fuente, no un evento real.
UMBRAL_EXTREMO = 0.50

# Una empresa real tiene a lo mucho uno o dos movimientos de esa magnitud en
# once anios. Mas que eso no es volatilidad, es un dato mal medido.
MAX_EVENTOS_EXTREMOS = 2

# Dias consecutivos sin que el precio cambie ni un centavo. Imposible para una
# accion de indice: o esta suspendida, o la fuente repite el ultimo dato.
DIAS_ESTANCADO = 10

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER}")

precios = spark.table(f"{BRONZE}.prices")
constituyentes = spark.table(f"{BRONZE}.constituents")

print(f"Bronze: {precios.count():,} filas, {precios.select('ticker').distinct().count():,} tickers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. La ventana: precio anterior y rendimiento
# MAGIC
# MAGIC `Window.partitionBy("ticker").orderBy("date")` define, para cada fila, un
# MAGIC "vecindario": las demás filas **del mismo ticker**, ordenadas por fecha.
# MAGIC `F.lag("close")` toma el valor de la fila anterior dentro de ese vecindario.
# MAGIC
# MAGIC El `partitionBy` es lo que evita el error clásico: sin él, el precio
# MAGIC anterior de la primera fila de Apple sería el último de Amazon.
# MAGIC
# MAGIC **Por qué rendimiento logarítmico y no el precio:** una serie de precios no
# MAGIC es estacionaria. Correlacionar los precios de dos empresas que subieron da
# MAGIC ~0.95 y eso no significa que se muevan juntas — significa que las dos
# MAGIC subieron. `ln(hoy/ayer)` resuelve eso.
# MAGIC
# MAGIC Nada de esto se ejecuta todavía: son *transformations*, y Spark solo las
# MAGIC anota. El trabajo real empieza en la primera *action*.

# COMMAND ----------

ventana = Window.partitionBy("ticker").orderBy("date")

con_rendimiento = (
    precios
    .withColumn("prev_close", F.lag("close").over(ventana))
    .withColumn(
        "log_return",
        F.when(
            (F.col("prev_close").isNotNull()) & (F.col("prev_close") > 0),
            F.log(F.col("close") / F.col("prev_close")),
        ),
    )
)

# La primera fila de cada ticker no tiene dia previo con que compararse.
con_rendimiento = con_rendimiento.filter(F.col("log_return").isNotNull())

print("Transformations registradas. Spark aun no ha leido nada.")
display(con_rendimiento.select("date", "ticker", "close", "prev_close", "log_return").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Diagnóstico por ticker
# MAGIC
# MAGIC Antes de analizar hay que auditar. Una matriz de correlación no avisa
# MAGIC cuando está calculada sobre datos rotos: devuelve números que parecen
# MAGIC razonables.
# MAGIC
# MAGIC Tres anomalías distintas, cada una con su tratamiento:
# MAGIC
# MAGIC 1. **Poca historia** — la empresa entró al índice hace poco. No es un error;
# MAGIC    simplemente no alcanza para estimar correlaciones.
# MAGIC 2. **Movimientos extremos repetidos** — splits mal aplicados por la fuente.
# MAGIC 3. **Precio estancado** — días seguidos sin que cambie ni un centavo.

# COMMAND ----------

# Historia disponible por ticker
historia = con_rendimiento.groupBy("ticker").agg(
    F.count("*").alias("n_obs"),
    F.min("date").alias("primera_fecha"),
    F.max("date").alias("ultima_fecha"),
)

# Eventos extremos: cuantos dias de +/-50% tiene cada ticker
extremos = (
    con_rendimiento
    .withColumn("es_extremo", (F.abs("log_return") > UMBRAL_EXTREMO).cast("int"))
    .groupBy("ticker")
    .agg(F.sum("es_extremo").alias("n_eventos_extremos"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precios estancados: *gaps and islands*
# MAGIC
# MAGIC Contar días sueltos sin cambio no sirve — lo que delata un dato muerto es
# MAGIC una **racha**. El patrón para medirla se llama *gaps and islands* y es un
# MAGIC clásico de las window functions:
# MAGIC
# MAGIC 1. Marcar cada día como "cambió" (0) o "no cambió" (1).
# MAGIC 2. Una **suma acumulada de los ceros** da un identificador que se mantiene
# MAGIC    constante mientras el precio no se mueve, y aumenta en cuanto se mueve.
# MAGIC 3. Agrupar por ese identificador: cada grupo es una racha, y su tamaño es
# MAGIC    su duración.

# COMMAND ----------

acumulada = Window.partitionBy("ticker").orderBy("date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)

rachas = (
    con_rendimiento
    .withColumn("sin_cambio", (F.col("close") == F.col("prev_close")).cast("int"))
    .withColumn(
        "grupo_racha",
        F.sum(F.when(F.col("sin_cambio") == 0, 1).otherwise(0)).over(acumulada),
    )
    .filter(F.col("sin_cambio") == 1)
    .groupBy("ticker", "grupo_racha")
    .agg(F.count("*").alias("largo"))
    .groupBy("ticker")
    .agg(F.max("largo").alias("dias_estancado"))
)

display(
    rachas.filter(F.col("dias_estancado") >= DIAS_ESTANCADO)
    .orderBy(F.desc("dias_estancado"))
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reporte consolidado
# MAGIC
# MAGIC Tres `join` para unir los diagnósticos con los metadatos de cada empresa.
# MAGIC Se usa `left` porque un ticker sin rachas detectadas debe seguir apareciendo
# MAGIC en el reporte, con `dias_estancado` nulo.

# COMMAND ----------

reporte = (
    historia
    .join(extremos, on="ticker", how="left")
    .join(rachas, on="ticker", how="left")
    .join(
        constituyentes.select("ticker", "company", "sector", "cap_tier"),
        on="ticker", how="left",
    )
    .fillna({"n_eventos_extremos": 0, "dias_estancado": 0})
)

reporte = reporte.withColumn(
    "motivo_exclusion",
    F.when(F.col("n_eventos_extremos") > MAX_EVENTOS_EXTREMOS, "datos_corruptos")
     .when(F.col("dias_estancado") >= DIAS_ESTANCADO, "precio_estancado")
     .otherwise(F.lit(None)),
)

reporte.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{SILVER}.quality_report"
)

display(
    reporte.groupBy("motivo_exclusion").count().orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. El intercambio: cuántas empresas vs cuánta historia
# MAGIC
# MAGIC Un **panel balanceado** solo conserva los días en que *todos* los tickers
# MAGIC tienen precio. Tiene una trampa que cuesta caro ignorar:
# MAGIC
# MAGIC > **La longitud del panel la impone el ticker sobreviviente más joven.**
# MAGIC
# MAGIC Si se admite una empresa que entró al índice en 2024, todas las demás
# MAGIC pierden la historia anterior a esa fecha. Por eso no se elige el umbral a
# MAGIC ojo: se mide.

# COMMAND ----------

dias_totales = con_rendimiento.select("date").distinct().count()
validos = reporte.filter(F.col("motivo_exclusion").isNull())

print(f"Días de mercado en el periodo: {dias_totales:,}\n")
print(f"{'umbral':>8} {'empresas':>10} {'% universo':>12}")
for umbral in [500, 1000, 1500, 2000, 2500, 2800, 2900, dias_totales]:
    n = validos.filter(F.col("n_obs") >= umbral).count()
    print(f"{umbral:>8} {n:>10,} {100*n/validos.count():>11.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decisión del umbral
# MAGIC
# MAGIC Con el S&P 500 solo, exigir historia casi completa costaba 9% del universo.
# MAGIC Con el 1500 el costo es mayor: las empresas chicas entran y salen del índice
# MAGIC con mucha más frecuencia, así que hay más rotación.
# MAGIC
# MAGIC **Ese sesgo hay que declararlo:** al exigir historia larga nos quedamos con
# MAGIC las small caps que *sobrevivieron* once años, que no son una muestra
# MAGIC representativa de las small caps. Es sesgo de supervivencia, y es más fuerte
# MAGIC aquí que en el Entregable 1.

# COMMAND ----------

MIN_OBSERVACIONES = 2500  # ajustar según la tabla de arriba

tickers_validos = (
    validos.filter(F.col("n_obs") >= MIN_OBSERVACIONES)
    .select("ticker")
)
n_validos = tickers_validos.count()

# Un panel balanceado conserva solo las fechas completas: aquellas en que
# cotizaron los n_validos tickers. No se rellena hacia adelante (ffill): eso
# inventaria dias con rendimiento 0% que nunca ocurrieron y sesgaria todas las
# correlaciones hacia abajo.
filtrado = con_rendimiento.join(F.broadcast(tickers_validos), on="ticker", how="inner")

fechas_completas = (
    filtrado.groupBy("date").agg(F.count("*").alias("n"))
    .filter(F.col("n") == n_validos)
    .select("date")
)

panel = filtrado.join(F.broadcast(fechas_completas), on="date", how="inner")

print(f"tickers conservados: {n_validos:,}")
print(f"fechas completas:    {fechas_completas.count():,} de {dias_totales:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Escritura de la capa Silver
# MAGIC
# MAGIC Se particiona por año, igual que Bronze: las consultas por periodo leen
# MAGIC solo los archivos de ese año en lugar de la tabla completa.

# COMMAND ----------

silver_returns = (
    panel
    .join(
        constituyentes.select("ticker", "sector", "cap_tier"),
        on="ticker", how="left",
    )
    .select(
        "date", "ticker", "close", "log_return",
        "sector", "cap_tier",
        F.year("date").alias("year"),
    )
)

(
    silver_returns.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("year")
    .format("delta")
    .saveAsTable(f"{SILVER}.returns")
)

resultado = spark.table(f"{SILVER}.returns")
print(f"silver.returns: {resultado.count():,} filas")
display(resultado.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificación

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   cap_tier,
# MAGIC   COUNT(DISTINCT ticker)        AS empresas,
# MAGIC   COUNT(*)                      AS observaciones,
# MAGIC   ROUND(AVG(log_return), 6)     AS rendimiento_medio,
# MAGIC   ROUND(STDDEV(log_return), 4)  AS volatilidad_diaria
# MAGIC FROM workspace.silver.returns
# MAGIC GROUP BY cap_tier
# MAGIC ORDER BY empresas DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué sigue
# MAGIC
# MAGIC `03_gold_analisis` toma `silver.returns` y calcula la estructura de
# MAGIC correlación a escala, más el factor de mercado por nivel de capitalización:
# MAGIC **¿dependen menos del mercado las empresas chicas que las grandes?**
