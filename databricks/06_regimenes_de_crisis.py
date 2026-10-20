# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — La diversificación muere en las crisis
# MAGIC
# MAGIC **Entregable 2 · Capa Gold**
# MAGIC
# MAGIC El motor de riesgo del notebook anterior tiene un supuesto escondido: calcula
# MAGIC **una sola** matriz de correlación para once años, como si la estructura del
# MAGIC mercado fuera constante.
# MAGIC
# MAGIC No lo es. Y la forma en que cambia es la peor posible.
# MAGIC
# MAGIC > **La correlación se dispara exactamente cuando el mercado cae.** La
# MAGIC > diversificación desaparece justo el día que la necesitas.
# MAGIC
# MAGIC Es contraintuitivo pero tiene una explicación simple: en calma, cada empresa
# MAGIC se mueve por sus propias razones —un contrato, un reporte trimestral, un
# MAGIC producto nuevo—. En pánico, esas razones dejan de importar porque todo el
# MAGIC mundo compra o vende todo al mismo tiempo.
# MAGIC
# MAGIC **Una precisión que los datos imponen:** lo que dispara la cohesión no es
# MAGIC que el mercado *baje*, es que se *agite*. La medición de la sección 2 lo
# MAGIC muestra — la cohesión correlaciona mucho más con la volatilidad del mes que
# MAGIC con su rendimiento. Abril de 2020 es el caso que lo delata: cohesión altísima
# MAGIC con un rendimiento fuertemente positivo, porque fue el rebote. Pánico y
# MAGIC euforia rompen la diversificación por igual.
# MAGIC
# MAGIC ## Cómo se mide sin calcular 733,000 pares por día
# MAGIC
# MAGIC Medir la correlación promedio día por día entre 1,211 acciones sería
# MAGIC imposible. El atajo es el mismo del notebook 03: en lugar de comparar cada
# MAGIC acción con todas las demás, se compara con **el mercado**.
# MAGIC
# MAGIC El **R² promedio de todas las acciones contra el factor de mercado** mide
# MAGIC exactamente lo que buscamos: qué fracción del movimiento de una acción típica
# MAGIC es "el mercado". Cuando sube, es que todo se mueve junto. Y se calcula con un
# MAGIC `groupBy` sobre el mes, no con un producto cartesiano.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Arranque: encontrar el repositorio
# MAGIC
# MAGIC Igual que el notebook 05: la matemática de riesgo vive en `src/risk.py`, un
# MAGIC nivel arriba de `databricks/`. Sin rutas escritas a mano.

# COMMAND ----------

import sys
from pathlib import Path


def encontrar_raiz_del_repo() -> Path:
    """Localiza la raíz del repositorio sin asumir ninguna ruta fija."""
    actual = Path.cwd()
    for candidato in [actual, *actual.parents]:
        if (candidato / "src" / "risk.py").is_file():
            return candidato
    try:
        ruta = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook().getContext().notebookPath().get()
        )
        base = Path("/Workspace") / ruta.lstrip("/")
        for candidato in base.parents:
            if (candidato / "src" / "risk.py").is_file():
                return candidato
    except Exception:
        pass
    raise RuntimeError(
        "No se encontró src/risk.py. Verifica que el notebook se ejecute desde "
        "el Git folder del repositorio."
    )


RAIZ = encontrar_raiz_del_repo()
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

print(f"Raíz del repositorio: {RAIZ}")

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from src.risk import (
    DIAS_HABILES,
    classify_regimes,
    regime_comparison,
)

CATALOGO = "workspace"
SILVER = f"{CATALOGO}.silver"
GOLD = f"{CATALOGO}.gold"
# Paleta del proyecto, validada para visión con daltonismo.
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_SUAVE = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": TINTA_SUAVE,
    "text.color": TINTA, "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
    "figure.facecolor": "white",
})

residuales = spark.table(f"{GOLD}.residual_returns")
factor = spark.table(f"{GOLD}.market_factor")
print(f"residuales: {residuales.count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cohesión del mercado, mes a mes
# MAGIC
# MAGIC Para cada mes y cada acción se calcula su correlación con el mercado durante
# MAGIC ese mes (~21 días). Luego se promedia entre las 1,211 acciones.
# MAGIC
# MAGIC El resultado es **una serie de tiempo de un solo número**: qué tan junto se
# MAGIC movió el mercado ese mes.

# COMMAND ----------

por_mes = (
    residuales
    .withColumn("mes", F.date_trunc("month", F.col("date")))
    .groupBy("mes", "ticker", "cap_tier")
    .agg(
        F.corr("log_return", "mkt_return").alias("corr_mercado"),
        F.count("*").alias("dias"),
    )
    # Meses con menos de 15 días hábiles dan correlaciones inestables.
    .filter(F.col("dias") >= 15)
    .withColumn("r2", F.pow("corr_mercado", 2))
)

cohesion = (
    por_mes.groupBy("mes")
    .agg(
        F.avg("r2").alias("cohesion"),
        F.avg("corr_mercado").alias("corr_media"),
        F.count("*").alias("n_acciones"),
    )
    .orderBy("mes")
)

(
    cohesion.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.market_cohesion")
)

df_cohesion = spark.table(f"{GOLD}.market_cohesion").toPandas()
df_cohesion["mes"] = pd.to_datetime(df_cohesion["mes"])
df_cohesion = df_cohesion.sort_values("mes").reset_index(drop=True)

print(f"{len(df_cohesion)} meses")
print(f"  cohesión mínima : {df_cohesion['cohesion'].min():.3f}  "
      f"({df_cohesion.loc[df_cohesion['cohesion'].idxmin(), 'mes']:%Y-%m})")
print(f"  cohesión máxima : {df_cohesion['cohesion'].max():.3f}  "
      f"({df_cohesion.loc[df_cohesion['cohesion'].idxmax(), 'mes']:%Y-%m})")
print(f"  mediana         : {df_cohesion['cohesion'].median():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Los meses de mayor y menor cohesión

# COMMAND ----------

print("MESES DE MAYOR COHESIÓN (todo se movía junto)")
for _, f in df_cohesion.nlargest(8, "cohesion").iterrows():
    print(f"  {f['mes']:%Y-%m}   R² promedio = {f['cohesion']:.3f}")

print("\nMESES DE MENOR COHESIÓN (cada quien por su lado)")
for _, f in df_cohesion.nsmallest(5, "cohesion").iterrows():
    print(f"  {f['mes']:%Y-%m}   R² promedio = {f['cohesion']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cohesión contra rendimiento del mercado
# MAGIC
# MAGIC La prueba central: ¿la cohesión sube cuando el mercado cae?
# MAGIC
# MAGIC Si la correlación entre ambas series es negativa, la respuesta es sí — y
# MAGIC significa que la diversificación se deteriora precisamente en los meses malos.

# COMMAND ----------

rendimiento_mes = (
    factor
    .withColumn("mes", F.date_trunc("month", F.col("date")))
    .groupBy("mes")
    .agg(
        F.sum("mkt_return").alias("rend_mes"),
        F.stddev("mkt_return").alias("vol_mes"),
    )
    .toPandas()
)
rendimiento_mes["mes"] = pd.to_datetime(rendimiento_mes["mes"])

datos = df_cohesion.merge(rendimiento_mes, on="mes").sort_values("mes")

corr_rend = datos["cohesion"].corr(datos["rend_mes"])
corr_vol = datos["cohesion"].corr(datos["vol_mes"])

print(f"  correlación (cohesión, rendimiento del mes) : {corr_rend:+.3f}")
print(f"  correlación (cohesión, volatilidad del mes) : {corr_vol:+.3f}")
print()

# classify_regimes parte los meses en tercios. Esta en src/risk.py con sus
# pruebas, en lugar de repetir la logica del quantile en cada notebook.
datos["regimen"] = classify_regimes(datos.set_index("mes")["rend_mes"]).values
tabla = datos.groupby("regimen").agg(
    meses=("cohesion", "size"),
    cohesion_media=("cohesion", "mean"),
    rend_medio=("rend_mes", "mean"),
).reindex(["meses malos", "meses normales", "meses buenos"])

print(f"{'régimen':18} {'meses':>7} {'cohesión':>11} {'rend. medio':>13}")
print("-" * 52)
for nombre, f in tabla.iterrows():
    print(f"{nombre:18} {int(f['meses']):>7} {f['cohesion_media']:>11.3f} "
          f"{100 * f['rend_medio']:>12.2f}%")

malos = tabla.loc["meses malos", "cohesion_media"]
buenos = tabla.loc["meses buenos", "cohesion_media"]
print()
print(f"En los meses malos la cohesión es {malos / buenos:.2f} veces la de los buenos.")
print()
print("PERO COMPARA LAS DOS CORRELACIONES DE ARRIBA.")
print(f"  con la volatilidad: {corr_vol:+.3f}")
print(f"  con el rendimiento: {corr_rend:+.3f}")
print()
print("La de volatilidad es bastante mas fuerte, y eso cambia la lectura: la")
print("cohesion no sigue las CAIDAS, sigue la TURBULENCIA. El caso que lo")
print("delata es abril de 2020, con cohesion altisima y rendimiento muy")
print("positivo: era el rebote posterior al desplome de COVID.")
print()
print("Pánico y euforia mueven todo junto por igual. Lo que rompe la")
print("diversificación no es que el mercado baje, es que se agite.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. La gráfica

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6.2), sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12},
)

ax1.plot(datos["mes"], datos["cohesion"], linewidth=1.8, color=AZUL)
ax1.axhline(datos["cohesion"].median(), color=TINTA_SUAVE,
            linewidth=1, linestyle="--")
ax1.set_ylabel("Cohesión del mercado\n(R² promedio)")
ax1.set_title("La diversificación desaparece justo cuando se necesita",
              fontsize=13, fontweight="bold", color=TINTA, loc="left", pad=12)

pico = datos.loc[datos["cohesion"].idxmax()]
ax1.annotate(
    f"{pico['mes']:%b %Y}: {pico['cohesion']:.2f}",
    xy=(pico["mes"], pico["cohesion"]),
    xytext=(14, -6), textcoords="offset points",
    fontsize=9, fontweight="bold", color=TINTA,
    arrowprops=dict(arrowstyle="-", color=TINTA_SUAVE, linewidth=0.9),
)

colores = [NARANJA if r < 0 else AZUL for r in datos["rend_mes"]]
ax2.bar(datos["mes"], 100 * datos["rend_mes"], width=22, color=colores)
ax2.axhline(0, color=TINTA_SUAVE, linewidth=0.8)
ax2.set_ylabel("Rendimiento\nmensual (%)")
ax2.set_xlabel("")

fig.text(0.125, -0.02,
         "Barras naranjas = meses de pérdida. Los picos de cohesión coinciden con "
         "los meses agitados, caigan o reboten.",
         fontsize=8.5, color=TINTA_SUAVE)

ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax2.grid(axis="x", visible=False)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ¿Protege el tamaño de la empresa?
# MAGIC
# MAGIC El notebook 03 encontró que las small caps dependen del mercado tanto como
# MAGIC las grandes, pero reaccionan más. La pregunta ahora es si ese patrón
# MAGIC **cambia en las crisis**: ¿se salvan las chicas cuando todo cae, o se
# MAGIC contagian igual?

# COMMAND ----------

cohesion_tier = (
    por_mes.groupBy("mes", "cap_tier")
    .agg(F.avg("r2").alias("cohesion"))
    .toPandas()
)
cohesion_tier["mes"] = pd.to_datetime(cohesion_tier["mes"])

pivote = (
    cohesion_tier.pivot(index="mes", columns="cap_tier", values="cohesion")
    .merge(datos[["mes", "regimen"]].set_index("mes"),
           left_index=True, right_index=True)
)

resumen_tier = pivote.groupby("regimen")[["large", "mid", "small"]].mean().reindex(
    ["meses malos", "meses normales", "meses buenos"]
)

print(f"{'régimen':18} {'large':>9} {'mid':>9} {'small':>9}")
print("-" * 48)
for nombre, f in resumen_tier.iterrows():
    print(f"{nombre:18} {f['large']:>9.3f} {f['mid']:>9.3f} {f['small']:>9.3f}")

print()
print("CUÁNTO SUBE LA COHESIÓN DE BUENOS A MALOS MESES")
for tier in ["large", "mid", "small"]:
    salto = resumen_tier.loc["meses malos", tier] / resumen_tier.loc["meses buenos", tier]
    print(f"  {tier:8} {salto:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Lo que esto significa para el motor de riesgo
# MAGIC
# MAGIC El notebook 05 reporta un número de riesgo. Este notebook demuestra que ese
# MAGIC número **depende del régimen**. Aquí se cuantifica la diferencia: el mismo
# MAGIC portafolio, con la covarianza estimada solo con meses tranquilos y solo con
# MAGIC meses de crisis.

# COMMAND ----------

ancho = (
    spark.table(f"{SILVER}.returns")
    .groupBy("date").pivot("ticker").agg(F.first("log_return"))
    .orderBy("date").toPandas().set_index("date")
).dropna(axis=1, how="any")
ancho.index = pd.to_datetime(ancho.index)

CANDIDATOS = ["JPM", "AAPL", "JNJ", "XOM", "PG", "NEE", "AMT", "LIN", "UNP", "VZ"]
cartera = [t for t in CANDIDATOS if t in ancho.columns]

# Los regímenes van indexados por periodo mensual, que es como los espera
# regime_comparison para cruzarlos con las fechas diarias.
regimenes = pd.Series(
    datos["regimen"].values,
    index=pd.PeriodIndex(datos["mes"], freq="M"),
)

tabla_regimen = regime_comparison(ancho, regimenes, cartera)

print(f"Portafolio de prueba: {len(cartera)} acciones de sectores distintos, pesos iguales")
print()
print(f"{'régimen':20} {'días':>7} {'vol portafolio':>16} {'vol individual':>16} {'ratio divers.':>14}")
print("-" * 78)
for _, f in tabla_regimen.iterrows():
    print(f"{f['regimen']:20} {int(f['dias']):>7} {100 * f['vol_portafolio']:>15.2f}% "
          f"{100 * f['vol_individual']:>15.2f}% {f['ratio_diversificacion']:>14.2f}")

print()
print("La columna que importa es la última. Si el ratio de diversificación baja")
print("en los meses malos, significa que las correlaciones subieron: las mismas")
print("diez acciones protegen menos justo cuando hace falta que protejan.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conclusión
# MAGIC
# MAGIC Las dos piezas del Entregable 2 se cierran una a la otra:
# MAGIC
# MAGIC - El **notebook 05** construye un motor que mide el riesgo de un portafolio.
# MAGIC - El **notebook 06** demuestra que ese riesgo **no es un número fijo**: la
# MAGIC   estructura de correlación que el motor usa cambia con el régimen del
# MAGIC   mercado, y cambia en la dirección desfavorable.
# MAGIC
# MAGIC La consecuencia práctica es incómoda: **un motor de riesgo calibrado con la
# MAGIC historia completa subestima el riesgo en las crisis**, que son justamente los
# MAGIC momentos en que alguien consulta un motor de riesgo.
# MAGIC
# MAGIC Esa es también la razón por la que las plataformas reales no usan una sola
# MAGIC matriz: corren escenarios de estrés con correlaciones de crisis. Dejar
# MAGIC planteado ese problema, con la medición que lo sustenta, es un buen punto de
# MAGIC partida para el Entregable 3.
