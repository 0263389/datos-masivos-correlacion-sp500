# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: el factor de mercado
# MAGIC
# MAGIC **Entregable 2 · Capa Gold · Responsable: Octavio de la Mora**
# MAGIC
# MAGIC El Entregable 1 dejó una pregunta abierta. Encontramos que las acciones del
# MAGIC mismo sector correlacionan 0.44 contra 0.32 entre sectores distintos — pero
# MAGIC también que **la correlación media entre sectores distintos no era cercana a
# MAGIC cero**, sino 0.32. Eso solo se explica de una forma: hay algo que mueve a
# MAGIC todas las acciones a la vez.
# MAGIC
# MAGIC Ese algo es el **factor de mercado**, y separarlo es el objetivo de esta capa.
# MAGIC
# MAGIC ## La pregunta
# MAGIC
# MAGIC > **¿Dependen menos del mercado las empresas chicas que las grandes?**
# MAGIC
# MAGIC Hay dos hipótesis plausibles y opuestas:
# MAGIC
# MAGIC - *Menos*: una small cap vive de su propio negocio, su destino es idiosincrático.
# MAGIC - *Más*: una small cap es frágil, y cuando el mercado cae, cae con todo.
# MAGIC
# MAGIC Con el S&P 500 solo esto no se podía responder: todas sus empresas son
# MAGIC grandes. Por eso ampliamos a 1,500.
# MAGIC
# MAGIC ## Por qué no se hace un self-join
# MAGIC
# MAGIC Correlacionar 1,200 tickers son ~720,000 pares. Un `join` de la tabla consigo
# MAGIC misma por fecha produciría 1,200 × 1,200 × 2,903 ≈ **4 mil millones de filas**
# MAGIC y reventaría la cuota de Free Edition.
# MAGIC
# MAGIC El factor de mercado evita eso: en lugar de comparar cada acción con todas
# MAGIC las demás, se compara con **un solo índice**. El costo pasa de O(n²) a O(n).

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOGO = "workspace"
SILVER = f"{CATALOGO}.silver"
GOLD = f"{CATALOGO}.gold"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD}")

rendimientos = spark.table(f"{SILVER}.returns")

n_tickers = rendimientos.select("ticker").distinct().count()
n_fechas = rendimientos.select("date").distinct().count()
print(f"Silver: {rendimientos.count():,} filas | {n_tickers:,} tickers | {n_fechas:,} fechas")
print(f"Pares que NO vamos a calcular: {n_tickers * (n_tickers - 1) // 2:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Construir el factor de mercado
# MAGIC
# MAGIC El factor es el **rendimiento promedio de todas las acciones cada día**: un
# MAGIC índice de igual ponderación. Si un día sube 1%, el mercado entero subió 1%.
# MAGIC
# MAGIC **Por qué este y no PCA.** El primer componente principal de la matriz de
# MAGIC rendimientos es, en la práctica, casi idéntico a este promedio: su vector
# MAGIC propio tiene todos los pesos del mismo signo y parecidos entre sí. Pero PCA
# MAGIC exige construir la matriz de covarianza de 1,200×1,200, mientras que esto es
# MAGIC un `groupBy` sobre la fecha. Mismo resultado, fracción del costo.
# MAGIC
# MAGIC Se usa igual ponderación —no ponderación por capitalización— porque el
# MAGIC objetivo es medir *el mercado como conjunto*, no dejar que cinco mega-caps
# MAGIC definan solas lo que llamamos "mercado".

# COMMAND ----------

factor_mercado = (
    rendimientos.groupBy("date")
    .agg(
        F.avg("log_return").alias("mkt_return"),
        F.count("*").alias("n_acciones"),
    )
)

# El factor se usa en varios joins mas adelante, asi que conviene no
# recalcularlo cada vez. La forma obvia seria .cache(), pero **el compute
# serverless no lo permite**: devuelve NOT_SUPPORTED_WITH_SERVERLESS, porque
# cachear reserva memoria en un cluster fijo y serverless no asigna uno.
#
# La alternativa es mejor que el parche: materializarlo como tabla Delta. El
# resultado queda guardado en el lake en lugar de en la memoria de una sesion,
# asi que sobrevive al notebook y lo pueden leer las demas capas.
(
    factor_mercado.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.market_factor")
)
factor_mercado = spark.table(f"{GOLD}.market_factor")

print(f"días del factor: {factor_mercado.count():,}")
display(factor_mercado.orderBy(F.desc("date")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### El factor contra la realidad
# MAGIC
# MAGIC Prueba de sentido común: los peores días del factor deberían coincidir con
# MAGIC crisis conocidas. Si no lo hacen, el factor está mal construido.

# COMMAND ----------

print("PEORES DÍAS DEL MERCADO")
for f in factor_mercado.orderBy("mkt_return").limit(8).collect():
    print(f"  {f['date']}   {100 * f['mkt_return']:+7.2f}%")

print("\nMEJORES DÍAS DEL MERCADO")
for f in factor_mercado.orderBy(F.desc("mkt_return")).limit(5).collect():
    print(f"  {f['date']}   {100 * f['mkt_return']:+7.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Beta y R²: cuánto explica el mercado a cada acción
# MAGIC
# MAGIC Se unen los rendimientos de cada acción con el del mercado ese día, y se
# MAGIC calculan dos cantidades por ticker:
# MAGIC
# MAGIC | Medida | Qué significa |
# MAGIC |---|---|
# MAGIC | **Beta** | Cuánto se mueve la acción cuando el mercado se mueve 1%. Beta 1.5 = amplifica; beta 0.5 = amortigua |
# MAGIC | **R²** | Qué **fracción** de su movimiento explica el mercado. Es el cuadrado de la correlación |
# MAGIC
# MAGIC Los dos responden cosas distintas y es fácil confundirlos. Una acción puede
# MAGIC tener beta alta y R² bajo: amplifica mucho al mercado los días que lo sigue,
# MAGIC pero casi siempre se mueve por razones propias. **Para nuestra pregunta el
# MAGIC que importa es R²**: mide dependencia, no amplificación.
# MAGIC
# MAGIC `F.corr`, `F.covar_samp` y `F.var_samp` son agregaciones nativas de Spark,
# MAGIC así que todo esto es un solo `groupBy`.

# COMMAND ----------

con_mercado = rendimientos.join(
    F.broadcast(factor_mercado.select("date", "mkt_return")),
    on="date", how="inner",
)

betas = (
    con_mercado.groupBy("ticker", "sector", "cap_tier")
    .agg(
        F.count("*").alias("n_obs"),
        F.corr("log_return", "mkt_return").alias("corr_mercado"),
        (
            F.covar_samp("log_return", "mkt_return")
            / F.var_samp("mkt_return")
        ).alias("beta"),
        F.stddev("log_return").alias("vol_diaria"),
    )
    .withColumn("r2", F.pow("corr_mercado", 2))
)

# Misma razon que arriba: en lugar de cache(), se escribe y se vuelve a leer.
betas.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{GOLD}.market_factor_betas"
)
betas = spark.table(f"{GOLD}.market_factor_betas")

print(f"{betas.count():,} tickers con beta y R² calculados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. La respuesta: dependencia del mercado por tamaño

# COMMAND ----------

resumen = (
    betas.groupBy("cap_tier")
    .agg(
        F.count("*").alias("empresas"),
        F.avg("beta").alias("beta_medio"),
        F.avg("r2").alias("r2_medio"),
        F.expr("percentile_approx(r2, 0.5)").alias("r2_mediana"),
        F.avg("vol_diaria").alias("vol_media"),
    )
    .orderBy(F.desc("r2_medio"))
    .collect()
)

print(f"{'tamanio':8} {'empresas':>9} {'beta':>8} {'R2 medio':>10} {'R2 mediana':>12} {'volatilidad':>12}")
print("-" * 64)
for f in resumen:
    print(f"{f['cap_tier']:8} {f['empresas']:>9,} {f['beta_medio']:>8.3f} "
          f"{f['r2_medio']:>10.3f} {f['r2_mediana']:>12.3f} {f['vol_media']:>12.4f}")

print()
print("Lectura: R2 = fracción del movimiento diario que explica el mercado.")
print("R2 de 0.40 significa que el 40% de lo que hace esa acción es 'el mercado'")
print("y el 60% restante es propio de la empresa.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por sector y tamaño
# MAGIC
# MAGIC El tamaño no es lo único que importa: un banco grande y una biotecnológica
# MAGIC grande no dependen del mercado de la misma forma.

# COMMAND ----------

tabla_sector = (
    betas.groupBy("sector")
    .pivot("cap_tier", ["large", "mid", "small"])
    .agg(F.round(F.avg("r2"), 3))
    .orderBy("sector")
)

display(tabla_sector)

print("R2 PROMEDIO POR SECTOR Y TAMAÑO")
for f in tabla_sector.collect():
    print(f"  {str(f['sector'])[:24]:26} large={f['large'] or 0:<6} "
          f"mid={f['mid'] or 0:<6} small={f['small'] or 0:<6}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Los extremos
# MAGIC
# MAGIC Las empresas que más y menos dependen del mercado. Prueba de sentido común
# MAGIC otra vez: arriba deberían aparecer bancos y financieras (son el mercado),
# MAGIC y abajo biotecnológicas y mineras de oro (viven de su propia suerte).

# COMMAND ----------

print("MÁS DEPENDIENTES DEL MERCADO (R² más alto)")
for f in betas.orderBy(F.desc("r2")).limit(10).collect():
    print(f"  {f['ticker']:7} {str(f['sector'])[:22]:24} {f['cap_tier']:6} "
          f"R²={f['r2']:.3f}  beta={f['beta']:.2f}")

print("\nMENOS DEPENDIENTES DEL MERCADO (R² más bajo)")
for f in betas.orderBy("r2").limit(10).collect():
    print(f"  {f['ticker']:7} {str(f['sector'])[:22]:24} {f['cap_tier']:6} "
          f"R²={f['r2']:.3f}  beta={f['beta']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Rendimientos residuales
# MAGIC
# MAGIC El residual es lo que queda de una acción **después de quitarle el mercado**:
# MAGIC
# MAGIC ```
# MAGIC residual = rendimiento − beta × rendimiento_del_mercado
# MAGIC ```
# MAGIC
# MAGIC Es la parte idiosincrática, la que de verdad pertenece a esa empresa. Sobre
# MAGIC los residuales se puede volver a preguntar lo del Entregable 1 —¿correlacionan
# MAGIC las del mismo sector?— y esta vez la respuesta **no está inflada por el
# MAGIC movimiento común del mercado**.

# COMMAND ----------

residuales = (
    con_mercado
    .join(F.broadcast(betas.select("ticker", "beta")), on="ticker", how="inner")
    .withColumn(
        "residual",
        F.col("log_return") - F.col("beta") * F.col("mkt_return"),
    )
    .select("date", "ticker", "sector", "cap_tier", "log_return",
            "mkt_return", "residual", F.year("date").alias("year"))
)

(
    residuales.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("year")
    .format("delta")
    .saveAsTable(f"{GOLD}.residual_returns")
)

print(f"gold.residual_returns: {spark.table(f'{GOLD}.residual_returns').count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificación del residual
# MAGIC
# MAGIC Si la descomposición está bien hecha, los residuales deben tener **mucha
# MAGIC menos correlación con el mercado** que los rendimientos originales. Si no es
# MAGIC así, el beta está mal estimado.

# COMMAND ----------

verificacion = spark.table(f"{GOLD}.residual_returns").select(
    F.corr("log_return", "mkt_return").alias("corr_original"),
    F.corr("residual", "mkt_return").alias("corr_residual"),
    F.stddev("log_return").alias("vol_original"),
    F.stddev("residual").alias("vol_residual"),
).collect()[0]

print(f"  correlación con el mercado, original : {verificacion['corr_original']:+.4f}")
print(f"  correlación con el mercado, residual : {verificacion['corr_residual']:+.4f}")
print(f"  volatilidad original                 : {verificacion['vol_original']:.4f}")
print(f"  volatilidad residual                 : {verificacion['vol_residual']:.4f}")
print()
print("La correlación residual debe quedar cerca de cero: significa que el")
print("factor de mercado ya absorbió todo lo que podía explicar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ¿Sobrevive el efecto sector al quitar el mercado?
# MAGIC
# MAGIC Esta es la pregunta del Entregable 1, hecha sobre los residuales.
# MAGIC
# MAGIC En lugar de los 720,000 pares, se usa un atajo que escala linealmente:
# MAGIC construir el **residual promedio de cada sector** cada día, y medir cuánto se
# MAGIC parece el residual de cada empresa al de su propio sector.
# MAGIC
# MAGIC Si después de quitar el mercado una empresa sigue siguiendo a su sector, el
# MAGIC vínculo sectorial es real y no era un reflejo del movimiento común.

# COMMAND ----------

residual_sector = (
    spark.table(f"{GOLD}.residual_returns")
    .groupBy("date", "sector")
    .agg(F.avg("residual").alias("residual_sector"))
)

con_sector = (
    spark.table(f"{GOLD}.residual_returns")
    .join(residual_sector, on=["date", "sector"], how="inner")
)

efecto_sector = (
    con_sector.groupBy("cap_tier")
    .agg(
        F.corr("residual", "residual_sector").alias("corr_con_su_sector"),
        F.corr("log_return", "mkt_return").alias("corr_con_mercado"),
        F.count("*").alias("observaciones"),
    )
    .orderBy("cap_tier")
    .collect()
)

print(f"{'tamanio':8} {'con el mercado':>16} {'con su sector':>16}")
print("-" * 42)
for f in efecto_sector:
    print(f"{f['cap_tier']:8} {f['corr_con_mercado']:>16.4f} {f['corr_con_su_sector']:>16.4f}")

print()
print("Primera columna: correlación del rendimiento crudo con el mercado.")
print("Segunda: correlación del RESIDUAL con el residual de su sector, o sea")
print("el vínculo sectorial que sobrevive después de quitar el mercado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tabla Gold final

# COMMAND ----------

# Delta no deja sobrescribir una tabla mientras se lee de ella misma, asi que
# el resumen por tamanio va a su propia tabla en lugar de anexarse a los betas.
(
    spark.createDataFrame(
        [
            (
                f["cap_tier"],
                int(f["observaciones"]),
                float(f["corr_con_mercado"]),
                float(f["corr_con_su_sector"]),
            )
            for f in efecto_sector
        ],
        ["cap_tier", "observaciones", "corr_con_mercado", "corr_con_su_sector"],
    )
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.sector_effect_by_tier")
)

for tabla in ["market_factor", "market_factor_betas",
              "residual_returns", "sector_effect_by_tier"]:
    print(f"  {tabla:22} {spark.table(f'{GOLD}.{tabla}').count():>12,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué sigue
# MAGIC
# MAGIC `04_benchmark` mide el mismo cálculo en pandas y en Spark para encontrar a
# MAGIC partir de qué volumen conviene Spark — que es la pregunta de fondo de la
# MAGIC materia.
