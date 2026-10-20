# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 — Motor de riesgo de portafolio
# MAGIC
# MAGIC **Entregable 2 · Capa Gold**
# MAGIC
# MAGIC Hasta aquí el proyecto **analizaba**. Este notebook usa una **herramienta**:
# MAGIC le das una lista de acciones con sus pesos y te dice el riesgo real de ese
# MAGIC portafolio, y de dónde viene. Es la capa analítica que está en el centro de
# MAGIC plataformas como Aladdin.
# MAGIC
# MAGIC ## La idea que casi nadie tiene presente
# MAGIC
# MAGIC > **El riesgo de un portafolio no es el promedio del riesgo de sus partes.**
# MAGIC
# MAGIC Diez acciones con 30% de volatilidad cada una no dan un portafolio de 30%.
# MAGIC Pueden dar 15% si se mueven de forma independiente, o 28% si se mueven todas
# MAGIC igual. La diferencia **no está en las acciones, está en sus correlaciones**.
# MAGIC
# MAGIC ## Dónde vive el código
# MAGIC
# MAGIC La matemática no está en este notebook: vive en **`src/risk.py`**, con
# MAGIC **19 pruebas automáticas** en `tests/test_risk.py`.
# MAGIC
# MAGIC Es el mismo principio del Entregable 1, y por las mismas razones: una función
# MAGIC con entrada y salida se puede probar, reutilizar y editar en paralelo; una
# MAGIC celda de notebook no. Aquí solo queda la narrativa y las llamadas.
# MAGIC
# MAGIC `src/risk.py` no importa pyspark ni `src/config.py` a propósito, así que
# MAGIC corre igual aquí que en una laptop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Arranque: encontrar el repositorio
# MAGIC
# MAGIC Este notebook vive en `databricks/` y el código en `src/`, un nivel arriba.
# MAGIC Python no lo encuentra solo, igual que pasaba en el notebook del Entregable 1.
# MAGIC
# MAGIC La celda prueba dos caminos en orden: subir por el árbol desde el directorio
# MAGIC actual, y si eso no funciona, derivar la ruta del propio notebook. No hay
# MAGIC ninguna ruta escrita a mano.

# COMMAND ----------

import sys
from pathlib import Path


def encontrar_raiz_del_repo() -> Path:
    """Localiza la raíz del repositorio sin asumir ninguna ruta fija."""
    actual = Path.cwd()
    for candidato in [actual, *actual.parents]:
        if (candidato / "src" / "risk.py").is_file():
            return candidato

    # En Databricks el directorio de trabajo no siempre es el del notebook.
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
        "No se encontró src/risk.py. Verifica que el notebook se esté "
        "ejecutando desde el Git folder del repositorio y no desde una copia "
        "suelta en el workspace."
    )


RAIZ = encontrar_raiz_del_repo()
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

print(f"Raíz del repositorio: {RAIZ}")

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from src.risk import (
    DIAS_HABILES,
    condition_number,
    equal_weights,
    ledoit_wolf_shrinkage,
    min_variance_weights,
    risk_decomposition,
    risk_report,
    sample_covariance,
)

CATALOGO = "workspace"
SILVER = f"{CATALOGO}.silver"
GOLD = f"{CATALOGO}.gold"

print("src.risk importado correctamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. De Spark a matriz
# MAGIC
# MAGIC El `pivot` convierte el formato largo en una matriz de fechas × tickers — la
# MAGIC operación inversa a la de Bronze. Aquí sí conviene, porque el álgebra lineal
# MAGIC necesita la forma matricial.
# MAGIC
# MAGIC **`toPandas()` es la frontera del proyecto.** Spark reduce 3.5 millones de
# MAGIC filas a unas decenas de megabytes; de ahí en adelante el trabajo es álgebra
# MAGIC lineal densa, que es lo que numpy hace bien y Spark hace mal.

# COMMAND ----------

ancho = (
    spark.table(f"{SILVER}.returns")
    .groupBy("date").pivot("ticker").agg(F.first("log_return"))
    .orderBy("date")
)

matriz = ancho.toPandas().set_index("date").dropna(axis=1, how="any")
matriz.index = pd.to_datetime(matriz.index)

sectores = (
    spark.table(f"{GOLD}.market_factor_betas")
    .select("ticker", "sector").toPandas()
    .set_index("ticker")["sector"].to_dict()
)

X = matriz.to_numpy(dtype=np.float64)
T, N = X.shape

print(f"matriz: {T:,} fechas x {N:,} tickers ({matriz.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")
print(f"parámetros de la covarianza : {N * (N + 1) // 2:,}")
print(f"observaciones disponibles   : {T:,}")
print(f"razón T/N                   : {T / N:.1f}   (idealmente > 10)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Por qué la covarianza muestral está rota
# MAGIC
# MAGIC El síntoma se mide con el **número de condición**: el cociente entre el mayor
# MAGIC y el menor valor propio. Dice cuánto amplifica los errores al invertir la
# MAGIC matriz — y para optimizar un portafolio **hay que invertirla**.
# MAGIC
# MAGIC | Condición | Qué significa |
# MAGIC |---|---|
# MAGIC | ~10 | Matriz sana |
# MAGIC | ~1,000 | Empieza a ser frágil |
# MAGIC | > 10⁶ | Numéricamente inservible: el resultado es ruido amplificado |

# COMMAND ----------

cov_muestral = sample_covariance(X)
cov_encogida, delta, _ = ledoit_wolf_shrinkage(X)

vp_m = np.linalg.eigvalsh(cov_muestral)
vp_e = np.linalg.eigvalsh(cov_encogida)

print(f"  intensidad óptima δ : {delta:.4f}")
print(f"  → {100 * delta:.1f}% objetivo estructurado, {100 * (1 - delta):.1f}% muestral")
print()
print(f"{'':28} {'muestral':>16} {'encogida':>16}")
print("-" * 62)
print(f"{'número de condición':28} {condition_number(cov_muestral):>16,.0f} "
      f"{condition_number(cov_encogida):>16,.0f}")
print(f"{'valor propio menor':28} {vp_m[0]:>16.2e} {vp_e[0]:>16.2e}")
print(f"{'valores propios negativos':28} {(vp_m < 0).sum():>16} {(vp_e < 0).sum():>16}")
print()
print(f"La condición mejoró {condition_number(cov_muestral) / condition_number(cov_encogida):,.0f} veces.")
print()
print("δ se deriva en forma cerrada, no se elige a mano: es el valor que")
print("minimiza el error cuadrático esperado. Los detalles están en el")
print("docstring de ledoit_wolf_shrinkage.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. El motor
# MAGIC
# MAGIC | Medida | Fórmula | Qué responde |
# MAGIC |---|---|---|
# MAGIC | **Volatilidad** | √(wᵀΣw) | ¿Cuánto se mueve el portafolio? |
# MAGIC | **Contribución al riesgo** | wᵢ·(Σw)ᵢ/σₚ | ¿Cuánto aporta cada posición? |
# MAGIC | **Ratio de diversificación** | (wᵀσ)/σₚ | ¿Cuánto riesgo cancelaron las correlaciones? |
# MAGIC
# MAGIC La contribución al riesgo es la pieza no obvia: **no es proporcional al
# MAGIC dinero invertido**. Una posición del 5% puede aportar el 30% del riesgo si es
# MAGIC volátil y está correlacionada con el resto. Y las contribuciones suman
# MAGIC exactamente la volatilidad del portafolio, así que el reparto es una
# MAGIC identidad, no una aproximación — `check_sum` lo verifica en cada llamada.

# COMMAND ----------

def analizar(tickers: list[str], titulo: str, pesos=None) -> dict:
    """Localiza los tickers en la matriz y descompone el riesgo."""
    presentes = [t for t in tickers if t in matriz.columns]
    faltantes = set(tickers) - set(presentes)
    if faltantes:
        print(f"(no están en el panel limpio: {sorted(faltantes)})")

    indices = [matriz.columns.get_loc(t) for t in presentes]
    sigma = cov_encogida[np.ix_(indices, indices)]

    resultado = risk_decomposition(sigma, weights=pesos, labels=presentes)
    risk_report(resultado, titulo, sectors=sectores)
    return resultado


# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 1 — El portafolio que *parece* diversificado
# MAGIC
# MAGIC Diez acciones, todas grandes y conocidas, en partes iguales. Cualquiera diría
# MAGIC que está diversificado: son diez empresas distintas.

# COMMAND ----------

r_bancos = analizar(
    ["JPM", "BAC", "WFC", "C", "GS", "MS", "PNC", "USB", "TFC", "RF"],
    "DIEZ BANCOS, PESOS IGUALES",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 2 — El mismo número de acciones, repartidas entre sectores

# COMMAND ----------

r_sectores = analizar(
    ["JPM", "AAPL", "JNJ", "XOM", "PG", "NEE", "AMT", "LIN", "UNP", "VZ"],
    "DIEZ SECTORES DISTINTOS, PESOS IGUALES",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### La comparación

# COMMAND ----------

print(f"{'':34} {'10 bancos':>14} {'10 sectores':>14}")
print("-" * 64)
for etiqueta, clave, formato in [
    ("volatilidad del portafolio", "portfolio_vol", "pct"),
    ("promedio individual", "weighted_avg_vol", "pct"),
    ("ratio de diversificación", "diversification_ratio", "num"),
]:
    a, b = r_bancos[clave], r_sectores[clave]
    if formato == "pct":
        print(f"{etiqueta:34} {100 * a:13.2f}% {100 * b:13.2f}%")
    else:
        print(f"{etiqueta:34} {a:14.2f} {b:14.2f}")

ahorro = 1 - r_sectores["portfolio_vol"] / r_bancos["portfolio_vol"]
print()
print(f"Mismo número de posiciones, mismo peso por posición, y el portafolio")
print(f"repartido entre sectores tiene {100 * ahorro:.0f}% menos volatilidad.")
print("La diferencia completa está en las correlaciones.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ¿Sirve de verdad el shrinkage?
# MAGIC
# MAGIC Todo lo anterior asume que la matriz encogida es mejor. Eso hay que
# MAGIC **demostrarlo**, y la única prueba válida es **fuera de muestra**.
# MAGIC
# MAGIC Con la primera parte de la historia se estima la covarianza y se construye el
# MAGIC portafolio de **mínima varianza** — el que el modelo cree que será el más
# MAGIC estable. Con la parte final, que el modelo nunca vio, se mide la volatilidad
# MAGIC que **realmente** tuvo.
# MAGIC
# MAGIC Se usa un subconjunto de 150 activos para que la matriz muestral sea al menos
# MAGIC invertible: con los 1,200 y la mitad de la historia simplemente no lo es, y
# MAGIC el experimento no tendría contra qué comparar.

# COMMAND ----------

rng = np.random.default_rng(42)
seleccion = sorted(rng.choice(matriz.columns, size=150, replace=False))
sub = matriz[seleccion].to_numpy(dtype=np.float64)

corte = int(len(sub) * 0.7)
entrena, prueba = sub[:corte], sub[corte:]

cov_m_tr = sample_covariance(entrena)
cov_e_tr, delta_sub, _ = ledoit_wolf_shrinkage(entrena)

print(f"  entrenamiento : {entrena.shape[0]:,} días x {entrena.shape[1]} activos")
print(f"  prueba        : {prueba.shape[0]:,} días (nunca vistos)")
print(f"  δ             : {delta_sub:.4f}")
print()
print(f"{'estrategia':24} {'vol REAL':>11} {'vol prometida':>15} {'brecha':>9} "
      f"{'pos. máx':>10} {'en corto':>10}")
print("-" * 82)

for nombre, w, cov_usada in [
    ("mínima var. muestral", min_variance_weights(cov_m_tr), cov_m_tr),
    ("mínima var. encogida", min_variance_weights(cov_e_tr), cov_e_tr),
    ("pesos iguales", equal_weights(len(seleccion)), cov_m_tr),
]:
    vol_real = float(np.std(prueba @ w) * np.sqrt(DIAS_HABILES))
    prometida = float(np.sqrt(w @ cov_usada @ w))
    print(f"{nombre:24} {100 * vol_real:10.2f}% {100 * prometida:14.2f}% "
          f"{100 * (vol_real - prometida):8.2f}pp {100 * np.abs(w).max():9.1f}% "
          f"{100 * w[w < 0].sum():9.1f}%")

print()
print("La 'vol prometida' es lo que el modelo creía que iba a pasar; la 'vol")
print("REAL' es lo que pasó. La brecha mide cuánto se engañó el modelo consigo")
print("mismo, y es ahí donde la matriz muestral falla más feo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Guardar en el lake

# COMMAND ----------

diagnostico = pd.DataFrame({
    "metrica": ["activos", "observaciones", "razon_T_N", "condicion_muestral",
                "condicion_encogida", "delta_shrinkage"],
    "valor": [float(N), float(T), T / N, condition_number(cov_muestral),
              condition_number(cov_encogida), float(delta)],
})

riesgo = pd.DataFrame({
    "portafolio": ["10_bancos"] * len(r_bancos["labels"])
                  + ["10_sectores"] * len(r_sectores["labels"]),
    "ticker": list(r_bancos["labels"]) + list(r_sectores["labels"]),
    "peso": list(r_bancos["weights"]) + list(r_sectores["weights"]),
    "vol_individual": list(r_bancos["standalone_vol"]) + list(r_sectores["standalone_vol"]),
    "pct_riesgo": list(r_bancos["pct_risk"]) + list(r_sectores["pct_risk"]),
})

for nombre, df in [("covariance_diagnostics", diagnostico), ("portfolio_risk", riesgo)]:
    (
        spark.createDataFrame(df)
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD}.{nombre}")
    )
    print(f"  {nombre:26} {spark.table(f'{GOLD}.{nombre}').count():>6,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué sigue
# MAGIC
# MAGIC `06_regimenes_de_crisis` responde la otra mitad: este motor calcula el riesgo
# MAGIC **promedio de once años**, pero la correlación no es estable. En los periodos
# MAGIC agitados se dispara, y con ella el riesgo de cualquier portafolio.
