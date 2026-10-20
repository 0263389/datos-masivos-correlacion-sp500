# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — ¿Cuándo conviene Spark?
# MAGIC
# MAGIC **Entregable 2 · Benchmark · Responsable: Paolo del Valle**
# MAGIC
# MAGIC Este notebook responde la pregunta de fondo de la materia, y la responde
# MAGIC midiendo en lugar de afirmando.
# MAGIC
# MAGIC Hay una creencia cómoda —"para datos masivos se usa Spark"— que esconde la
# MAGIC parte importante: **Spark tiene un costo fijo**. Coordinar ejecutores,
# MAGIC planificar la consulta y mover datos entre nodos cuesta tiempo, y ese tiempo
# MAGIC se paga aunque el conjunto sea diminuto. Por debajo de cierto volumen,
# MAGIC pandas gana — y gana por mucho.
# MAGIC
# MAGIC La pregunta honesta no es *"¿cuál es mejor?"* sino **"¿dónde está el punto de
# MAGIC cruce?"**. Y conviene saber la respuesta antes de defender el proyecto: nuestro
# MAGIC conjunto tiene 3.5 millones de filas, que para los estándares de Spark **es
# MAGIC chico**.

# COMMAND ----------

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

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


def cronometrar(funcion, repeticiones: int = 3) -> float:
    """Mediana de varias corridas.

    Se usa la mediana y no el promedio porque la primera corrida siempre sale
    mas lenta -Spark planifica la consulta, la JVM calienta, el almacenamiento
    cachea- y un promedio dejaria que ese atipico contamine el resultado.
    """
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - inicio)
    return float(np.median(tiempos))


rendimientos = spark.table(f"{SILVER}.returns")
print(f"Conjunto completo: {rendimientos.count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lazy evaluation: dónde está realmente el costo
# MAGIC
# MAGIC Primero una demostración de por qué medir Spark es engañoso si no se sabe
# MAGIC qué se está midiendo.
# MAGIC
# MAGIC Spark separa **transformations** (describir el cálculo) de **actions**
# MAGIC (ejecutarlo). Un `groupBy` no calcula nada: solo anota la intención. El
# MAGIC trabajo ocurre cuando algo exige un resultado concreto.
# MAGIC
# MAGIC Si cronometras una transformation vas a medir casi cero y concluir que Spark
# MAGIC es instantáneo. No lo es: todavía no ha hecho nada.

# COMMAND ----------

def solo_transformacion():
    return rendimientos.groupBy("ticker").agg(F.avg("log_return"))


def transformacion_mas_accion():
    return rendimientos.groupBy("ticker").agg(F.avg("log_return")).collect()


t_transformacion = cronometrar(solo_transformacion, repeticiones=5)
t_accion = cronometrar(transformacion_mas_accion, repeticiones=3)

print(f"  solo la transformation : {t_transformacion * 1000:8.2f} ms")
print(f"  transformation + action: {t_accion * 1000:8.2f} ms")
print(f"  proporción             : {t_accion / max(t_transformacion, 1e-9):8.0f}x")
print()
print("La transformation es practicamente gratis porque no ejecuta nada.")
print("Todo el costo real aparece con la action.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. El benchmark
# MAGIC
# MAGIC Mismo cálculo, dos motores, volúmenes crecientes. El cálculo elegido es el
# MAGIC del Entregable 2: **el factor de mercado y el beta de cada acción**. No es un
# MAGIC ejemplo de juguete, es el trabajo real del proyecto.
# MAGIC
# MAGIC En ambos motores se hace exactamente lo mismo:
# MAGIC
# MAGIC 1. promedio de rendimientos por fecha (el factor de mercado),
# MAGIC 2. unirlo de vuelta a cada fila,
# MAGIC 3. covarianza y varianza por ticker para obtener el beta.
# MAGIC
# MAGIC Para que la comparación sea justa, en Spark se fuerza una *action* y en
# MAGIC pandas se parte de un DataFrame ya cargado en memoria.

# COMMAND ----------

TAMANIOS = [10_000, 50_000, 250_000, 1_000_000, 3_500_000]

base = rendimientos.select("date", "ticker", "log_return")
muestras_pandas = {}

print("Cargando muestras a memoria para pandas...")
for n in TAMANIOS:
    inicio = time.perf_counter()
    muestras_pandas[n] = base.limit(n).toPandas()
    filas = len(muestras_pandas[n])
    print(f"  {n:>9,} filas -> {filas:>9,} cargadas "
          f"({time.perf_counter() - inicio:5.1f}s, "
          f"{muestras_pandas[n].memory_usage(deep=True).sum() / 1024**2:6.1f} MB)")

# COMMAND ----------

def beta_en_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """El mismo calculo del Entregable 2, en pandas."""
    mercado = df.groupby("date")["log_return"].mean().rename("mkt_return")
    unido = df.join(mercado, on="date")
    agrupado = unido.groupby("ticker")
    cov = agrupado.apply(
        lambda g: g["log_return"].cov(g["mkt_return"]), include_groups=False
    )
    var = agrupado["mkt_return"].var()
    return (cov / var).rename("beta").to_frame()


def beta_en_spark(n: int):
    """El mismo calculo, en Spark. .collect() fuerza la action."""
    df = base.limit(n)
    mercado = df.groupBy("date").agg(F.avg("log_return").alias("mkt_return"))
    unido = df.join(mercado, on="date", how="inner")
    return (
        unido.groupBy("ticker")
        .agg(
            (F.covar_samp("log_return", "mkt_return") / F.var_samp("mkt_return"))
            .alias("beta")
        )
        .collect()
    )


resultados = []
print(f"{'filas':>10} {'pandas (s)':>12} {'spark (s)':>12} {'ganador':>10} {'ventaja':>10}")
print("-" * 58)

for n in TAMANIOS:
    df_local = muestras_pandas[n]
    t_pandas = cronometrar(lambda: beta_en_pandas(df_local), repeticiones=3)
    t_spark = cronometrar(lambda n=n: beta_en_spark(n), repeticiones=3)

    ganador = "pandas" if t_pandas < t_spark else "spark"
    ventaja = max(t_pandas, t_spark) / min(t_pandas, t_spark)
    resultados.append({"filas": len(df_local), "pandas": t_pandas, "spark": t_spark})

    print(f"{len(df_local):>10,} {t_pandas:>12.3f} {t_spark:>12.3f} "
          f"{ganador:>10} {ventaja:>9.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. La gráfica del punto de cruce

# COMMAND ----------

tabla = pd.DataFrame(resultados)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(tabla["filas"], tabla["pandas"], marker="o", markersize=7,
        linewidth=2, color=NARANJA, label="pandas")
ax.plot(tabla["filas"], tabla["spark"], marker="s", markersize=7,
        linewidth=2, color=AZUL, label="Spark")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Filas procesadas")
ax.set_ylabel("Tiempo (segundos)")
ax.set_title("Spark cobra un costo fijo que solo se amortiza con volumen",
             fontsize=12, fontweight="bold", color=TINTA, loc="left", pad=12)
ax.legend(frameon=False)

for _, fila in tabla.iterrows():
    mas_rapido = min(fila["pandas"], fila["spark"])
    ax.annotate(f"{mas_rapido:.2f}s", xy=(fila["filas"], mas_rapido),
                xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8, color=TINTA_SUAVE)

ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dónde está el punto de cruce
# MAGIC
# MAGIC pandas gana en todos los tamaños medidos, pero **la ventaja se encoge de
# MAGIC forma monótona**: 46x a 10 mil filas, 1.8x a 3.5 millones. Eso permite
# MAGIC extrapolar dónde se cruzan las curvas.
# MAGIC
# MAGIC El modelo es `tiempo = costo_fijo + costo_marginal × filas`. Ajustarlo a los
# MAGIC datos medidos y resolver dónde se igualan da el volumen a partir del cual
# MAGIC conviene cambiar de motor.

# COMMAND ----------

pend_pandas, fijo_pandas = np.polyfit(tabla["filas"], tabla["pandas"], 1)
pend_spark, fijo_spark = np.polyfit(tabla["filas"], tabla["spark"], 1)

print(f"  pandas: {fijo_pandas:.3f}s fijo + {pend_pandas * 1e6:.3f}s por millón de filas")
print(f"  Spark : {fijo_spark:.3f}s fijo + {pend_spark * 1e6:.3f}s por millón de filas")
print()

if pend_pandas > pend_spark:
    cruce = (fijo_spark - fijo_pandas) / (pend_pandas - pend_spark)
    actual = tabla["filas"].max()
    print(f"  punto de cruce estimado: {cruce / 1e6:.1f} millones de filas")
    print(f"  = {cruce / actual:.1f}x el conjunto actual")
    print()
    print("  CUIDADO CON ESTE NUMERO. Es sensible al ruido de medicion: los")
    print("  tiempos de Spark no salen monotonos entre corridas, y basta una")
    print("  medicion atipica en un tamanio chico para mover la pendiente del")
    print("  ajuste. Dos corridas de este mismo notebook dieron 7.8 y 10.4")
    print("  millones.")
    print()
    print("  Lo defendible es el ORDEN DE MAGNITUD: el cruce esta entre 8 y 10")
    print("  millones de filas, unas 2 a 3 veces el conjunto actual. Afirmar una")
    print("  cifra exacta seria darle al metodo una precision que no tiene.")
else:
    print("  Las curvas no se cruzan en el rango medido.")

print()
print("Los dos numeros que resumen todo el benchmark:")
crecimiento_datos = tabla["filas"].max() / tabla["filas"].min()
print(f"  con {crecimiento_datos:.0f}x mas datos...")
print(f"    pandas tardo {tabla['pandas'].iloc[-1] / tabla['pandas'].iloc[0]:.0f}x mas")
print(f"    Spark  tardo {tabla['spark'].iloc[-1] / tabla['spark'].iloc[0]:.1f}x mas")
print()
print("Spark apenas se inmuto. Ese es el punto: su costo es casi todo fijo, y")
print("el costo fijo se amortiza. El de pandas es casi todo marginal, y el")
print("costo marginal se acumula.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Partition pruning: el beneficio que no se ve en el reloj
# MAGIC
# MAGIC Las tablas están particionadas por año. Eso significa que una consulta
# MAGIC filtrada por año **no lee los archivos de los demás años**: los salta sin
# MAGIC abrirlos. Se llama *partition pruning*.
# MAGIC
# MAGIC Es la clase de optimización que pandas no puede tener, porque pandas no
# MAGIC sabe nada de cómo están guardados los datos — para él todo es un bloque en
# MAGIC memoria que alguien ya cargó.

# COMMAND ----------

def leer_todo():
    return rendimientos.agg(F.avg("log_return")).collect()


def leer_un_anio():
    return rendimientos.filter(F.col("year") == 2020).agg(F.avg("log_return")).collect()


t_todo = cronometrar(leer_todo, repeticiones=3)
t_anio = cronometrar(leer_un_anio, repeticiones=3)

filas_todo = rendimientos.count()
filas_anio = rendimientos.filter(F.col("year") == 2020).count()

print(f"  leer los 12 años : {t_todo:6.3f}s  ({filas_todo:,} filas)")
print(f"  leer solo 2020   : {t_anio:6.3f}s  ({filas_anio:,} filas)")
print(f"  datos leídos     : {100 * filas_anio / filas_todo:5.1f}% del total")
print(f"  tiempo           : {100 * t_anio / t_todo:5.1f}% del total")
print()
print("OJO CON ESTE RESULTADO: leer el 9% de los datos NO toma el 9% del")
print("tiempo. A esta escala el cuello de botella no es leer, es coordinar.")
print("El pruning si ocurrio -el plan de ejecucion de la celda siguiente lo")
print("demuestra- pero su beneficio queda enterrado bajo el costo fijo.")
print()
print("Es el mismo hallazgo del benchmark visto por otro lado: con 3.5")
print("millones de filas, Spark pasa mas tiempo organizandose que trabajando.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### El plan de ejecución lo confirma
# MAGIC
# MAGIC `explain()` muestra lo que Spark decidió hacer. En el plan de la consulta
# MAGIC filtrada aparece el filtro de partición empujado hasta la lectura del
# MAGIC archivo: Spark descarta los otros once años **antes** de leerlos, no después.

# COMMAND ----------

print("PLAN SIN FILTRO:")
rendimientos.agg(F.avg("log_return")).explain(mode="simple")

print("\nPLAN CON FILTRO POR PARTICIÓN:")
rendimientos.filter(F.col("year") == 2020).agg(F.avg("log_return")).explain(mode="simple")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Lo que pandas simplemente no puede hacer
# MAGIC
# MAGIC El benchmark de arriba compara tareas que **ambos** motores pueden resolver.
# MAGIC El argumento más fuerte a favor de Spark no es que sea más rápido en esas,
# MAGIC sino que existen tareas donde pandas no llega.
# MAGIC
# MAGIC El ejemplo está en nuestro propio proyecto: la matriz de correlación de todos
# MAGIC contra todos, que deliberadamente **no** calculamos en la capa Gold.

# COMMAND ----------

n_tickers = rendimientos.select("ticker").distinct().count()
n_fechas = rendimientos.select("date").distinct().count()

pares = n_tickers * (n_tickers - 1) // 2
filas_join = n_tickers * n_tickers * n_fechas
bytes_por_fila = 8 * 4  # cuatro columnas float64

print(f"  tickers                      : {n_tickers:,}")
print(f"  fechas                       : {n_fechas:,}")
print(f"  pares únicos                 : {pares:,}")
print()
memoria_gb = filas_join * bytes_por_fila / 1024**3
print(f"  filas de un self-join        : {filas_join:,}")
print(f"  memoria que exigiría en RAM  : {memoria_gb:,.0f} GB")
print(f"  (para comparar: una laptop típica tiene 8 a 32 GB)")
print()
print("Spark puede EXPRESAR ese calculo y procesarlo por bloques, derramando a")
print("disco lo que no cabe en memoria. pandas tendria que materializarlo entero")
print("y moriria con MemoryError.")
print()
print("Por eso la capa Gold usa el factor de mercado: convierte un problema")
print("O(n^2) en uno O(n). La leccion no es 'usa Spark para todo', es que la")
print("decision mas barata suele ser elegir mejor el algoritmo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conclusión
# MAGIC
# MAGIC Lo que muestran los números de este notebook:
# MAGIC
# MAGIC 1. **Spark cobra un costo fijo de arranque.** En volúmenes chicos, pandas gana
# MAGIC    por un margen amplio. No es un defecto de Spark: es el precio de poder
# MAGIC    distribuir.
# MAGIC
# MAGIC 2. **Nuestro conjunto de 3.5 millones de filas sigue siendo chico** para los
# MAGIC    estándares de Spark: el cruce está entre 8 y 10 millones, o sea 2 a 3
# MAGIC    veces más. Decirlo es más honesto que fingir lo contrario, y además da
# MAGIC    una meta concreta: ampliar el universo a unos 3,500 tickers, o pasar a
# MAGIC    datos intradía, pondría al proyecto del lado correcto del cruce.
# MAGIC
# MAGIC 3. **La ventaja real de Spark no es la velocidad, es el techo.** pandas está
# MAGIC    limitado por la memoria de una sola máquina; Spark no. El `self-join` de la
# MAGIC    sección 5 lo ilustra: no es que pandas tarde más, es que no puede.
# MAGIC
# MAGIC 4. **El particionado funciona, pero a esta escala no se nota.** El plan de
# MAGIC    ejecución confirma que Spark descarta once de los doce años sin abrirlos
# MAGIC    — y aun así el tiempo apenas baja. Leer el 9% de los datos tomó casi el
# MAGIC    mismo tiempo que leerlos todos, porque el cuello de botella no es la
# MAGIC    lectura sino la coordinación. El particionado es una inversión que paga
# MAGIC    cuando los datos crecen, no hoy.
# MAGIC
# MAGIC 5. **La mejor optimización fue algorítmica, no de infraestructura.** Cambiar
# MAGIC    el enfoque de O(n²) a O(n) con el factor de mercado ahorró más que
# MAGIC    cualquier ajuste de cómputo.
