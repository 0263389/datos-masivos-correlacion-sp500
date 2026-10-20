# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Motor de riesgo de portafolio
# MAGIC
# MAGIC **Entregable 2 · Capa Gold**
# MAGIC
# MAGIC Hasta aquí el proyecto **analizaba**. Este notebook construye una
# MAGIC **herramienta**: le das una lista de acciones con sus pesos y te dice el
# MAGIC riesgo real de ese portafolio, y de dónde viene.
# MAGIC
# MAGIC Es la capa analítica que está en el centro de plataformas como Aladdin.
# MAGIC
# MAGIC ## La idea que casi nadie tiene presente
# MAGIC
# MAGIC > **El riesgo de un portafolio no es el promedio del riesgo de sus partes.**
# MAGIC
# MAGIC Veinte acciones con 25% de volatilidad cada una no dan un portafolio de 25%.
# MAGIC Pueden dar 8% si se mueven de forma independiente, o 25% si se mueven todas
# MAGIC igual. La diferencia **no está en las acciones, está en sus correlaciones**.
# MAGIC
# MAGIC Por eso este motor necesita la matriz de covarianza, y por eso el proyecto
# MAGIC entero gira alrededor de estimarla bien.
# MAGIC
# MAGIC ## El patrón "pandas solo al final"
# MAGIC
# MAGIC Spark reduce 3.5 millones de filas a una matriz de 2,903 × 1,211 — unos 28 MB.
# MAGIC A partir de ahí el trabajo es **álgebra lineal densa**, que es justo lo que
# MAGIC numpy hace bien y Spark hace mal. La frontera entre los dos motores va
# MAGIC exactamente ahí.

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOGO = "workspace"
SILVER = f"{CATALOGO}.silver"
GOLD = f"{CATALOGO}.gold"

DIAS_HABILES = 252  # dias de mercado en un anio, para anualizar

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. De Spark a matriz
# MAGIC
# MAGIC El `pivot` convierte el formato largo en una matriz de fechas × tickers. Es
# MAGIC la operación inversa a la que hicimos en Bronze, y aquí sí conviene: el
# MAGIC álgebra lineal necesita la forma matricial.

# COMMAND ----------

rendimientos = spark.table(f"{SILVER}.returns")

ancho = (
    rendimientos.groupBy("date")
    .pivot("ticker")
    .agg(F.first("log_return"))
    .orderBy("date")
)

# toPandas() es la frontera: de aqui en adelante todo es numpy.
matriz = ancho.toPandas().set_index("date")
matriz = matriz.dropna(axis=1, how="any")

metadatos = (
    spark.table(f"{GOLD}.market_factor_betas")
    .select("ticker", "sector", "cap_tier")
    .toPandas()
    .set_index("ticker")
)

print(f"matriz de rendimientos: {matriz.shape[0]:,} fechas x {matriz.shape[1]:,} tickers")
print(f"memoria: {matriz.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"parámetros de la covarianza: {matriz.shape[1] * (matriz.shape[1] + 1) // 2:,}")
print(f"observaciones disponibles:   {matriz.shape[0]:,}")
print()
print("Ese desbalance es el problema central de esta capa, y es el tema de la")
print("siguiente sección.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Por qué la covarianza muestral está rota
# MAGIC
# MAGIC Con 1,211 activos hay **734,166 parámetros** que estimar y solo **2,903
# MAGIC observaciones**. Hay 250 veces más incógnitas que datos.
# MAGIC
# MAGIC El síntoma se mide con el **número de condición**: el cociente entre el mayor
# MAGIC y el menor valor propio de la matriz. Dice cuánto amplifica los errores al
# MAGIC invertirla — y para optimizar un portafolio **hay que invertirla**.
# MAGIC
# MAGIC | Condición | Qué significa |
# MAGIC |---|---|
# MAGIC | ~10 | Matriz sana |
# MAGIC | ~1,000 | Empieza a ser frágil |
# MAGIC | > 10⁶ | Numéricamente inservible: el resultado es ruido amplificado |
# MAGIC
# MAGIC Cuando hay más activos que observaciones la matriz ni siquiera es
# MAGIC invertible. Aquí no llegamos a ese extremo, pero sí a uno muy malo.

# COMMAND ----------

X = matriz.to_numpy(dtype=np.float64)
T, N = X.shape

cov_muestral = np.cov(X, rowvar=False) * DIAS_HABILES

valores_propios = np.linalg.eigvalsh(cov_muestral)
condicion = valores_propios[-1] / max(valores_propios[0], 1e-20)

print(f"  activos (N)                : {N:,}")
print(f"  observaciones (T)          : {T:,}")
print(f"  razón T/N                  : {T / N:.1f}   (idealmente > 10)")
print()
print(f"  valor propio mayor         : {valores_propios[-1]:.6f}")
print(f"  valor propio menor         : {valores_propios[0]:.2e}")
print(f"  NÚMERO DE CONDICIÓN        : {condicion:,.0f}")
print()
print(f"  valores propios negativos  : {(valores_propios < 0).sum()}")
print("  (deberían ser cero: una covarianza real no puede tenerlos. Los que")
print("   aparecen son error numérico puro.)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Shrinkage de Ledoit-Wolf
# MAGIC
# MAGIC La solución estándar es **encoger** la matriz muestral hacia una matriz
# MAGIC estructurada y estable:
# MAGIC
# MAGIC ```
# MAGIC Σ = δ · F  +  (1 − δ) · S
# MAGIC      ↑           ↑
# MAGIC   objetivo    muestral
# MAGIC   estable     insesgada
# MAGIC   pero rígido pero ruidosa
# MAGIC ```
# MAGIC
# MAGIC El objetivo `F` que se usa en finanzas es el de **correlación constante**:
# MAGIC conserva la volatilidad real de cada acción pero asume que *todos* los pares
# MAGIC tienen la misma correlación, el promedio observado. Es rígido, pero tiene
# MAGIC muy poco ruido porque estima un solo parámetro en vez de 733,000.
# MAGIC
# MAGIC Lo elegante del método de Ledoit y Wolf (2003) es que **δ no se elige a
# MAGIC mano**: se deriva en forma cerrada el valor que minimiza el error esperado.
# MAGIC Depende de cuánto ruido tiene la muestral (π) frente a cuánto se equivoca el
# MAGIC objetivo (γ).

# COMMAND ----------

def shrinkage_ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Covarianza encogida hacia el objetivo de correlacion constante.

    Implementa Ledoit & Wolf (2003), "Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection".

    Returns
    -------
    (covarianza encogida, delta optimo, objetivo F)
    """
    T, N = X.shape
    Y = X - X.mean(axis=0)              # desviaciones respecto a la media
    S = (Y.T @ Y) / T                   # covarianza muestral (sesgada, como en el paper)

    desv = np.sqrt(np.diag(S))
    corr = S / np.outer(desv, desv)

    # Correlacion promedio de los pares unicos: el unico parametro del objetivo.
    triangulo = np.triu_indices(N, k=1)
    r_media = float(np.mean(corr[triangulo]))

    # F: misma volatilidad individual, correlacion comun para todos los pares.
    F = r_media * np.outer(desv, desv)
    np.fill_diagonal(F, np.diag(S))

    # pi: cuanto ruido tiene cada entrada de la covarianza muestral.
    Y2 = Y ** 2
    pi_matriz = (Y2.T @ Y2) / T - S ** 2
    pi = float(pi_matriz.sum())

    # rho: parte del ruido que el objetivo tambien comparte, asi que no cuenta.
    # Los terminos cruzados usan theta, que mide como covaria el error de la
    # varianza de un activo con el error de su covarianza con otro.
    Y3 = Y ** 3
    theta = (Y3.T @ Y) / T - np.diag(S)[:, None] * S

    rho_diagonal = float(np.diag(pi_matriz).sum())
    razon = np.outer(desv, 1.0 / desv)
    cruzados = (r_media / 2.0) * (razon * theta.T + razon.T * theta)
    np.fill_diagonal(cruzados, 0.0)
    rho = rho_diagonal + float(cruzados.sum())

    # gamma: que tan lejos esta el objetivo de la muestral.
    gamma = float(((F - S) ** 2).sum())

    # delta optimo, acotado al intervalo valido.
    delta = max(0.0, min(1.0, ((pi - rho) / gamma) / T)) if gamma > 0 else 0.0

    return delta * F + (1 - delta) * S, delta, F


cov_encogida_diaria, delta, objetivo = shrinkage_ledoit_wolf(X)
cov_encogida = cov_encogida_diaria * DIAS_HABILES

vp_encogida = np.linalg.eigvalsh(cov_encogida)
condicion_encogida = vp_encogida[-1] / max(vp_encogida[0], 1e-20)

print(f"  intensidad óptima δ        : {delta:.4f}")
print(f"  → {100 * delta:.1f}% objetivo estructurado, {100 * (1 - delta):.1f}% muestral")
print()
print(f"{'':28} {'muestral':>16} {'encogida':>16}")
print("-" * 62)
print(f"{'número de condición':28} {condicion:>16,.0f} {condicion_encogida:>16,.0f}")
print(f"{'valor propio menor':28} {valores_propios[0]:>16.2e} {vp_encogida[0]:>16.2e}")
print(f"{'valores propios negativos':28} {(valores_propios < 0).sum():>16} {(vp_encogida < 0).sum():>16}")
print()
print(f"La condición mejoró {condicion / condicion_encogida:,.0f} veces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. El motor
# MAGIC
# MAGIC Con una matriz de covarianza estable ya se puede responder lo que importa.
# MAGIC Cuatro cantidades, cada una con su fórmula:
# MAGIC
# MAGIC | Medida | Fórmula | Qué responde |
# MAGIC |---|---|---|
# MAGIC | **Volatilidad** | √(wᵀΣw) | ¿Cuánto se mueve el portafolio? |
# MAGIC | **Contribución al riesgo** | wᵢ · (Σw)ᵢ / σₚ | ¿Cuánto aporta cada posición? |
# MAGIC | **Ratio de diversificación** | (wᵀσ) / σₚ | ¿Cuánto riesgo cancelaron las correlaciones? |
# MAGIC | **Volatilidad standalone** | √Σᵢᵢ | ¿Cuánto se movería sola? |
# MAGIC
# MAGIC La contribución al riesgo es la pieza no obvia. **No es proporcional al
# MAGIC dinero invertido**: una posición del 5% puede aportar el 30% del riesgo si es
# MAGIC volátil y está correlacionada con el resto. Las contribuciones suman
# MAGIC exactamente la volatilidad del portafolio, así que el reparto es exacto, no
# MAGIC una aproximación.
# MAGIC
# MAGIC El ratio de diversificación tiene una lectura directa: **cuántas posiciones
# MAGIC verdaderamente independientes tienes**. Un ratio de 1.0 significa que tus 20
# MAGIC acciones se comportan como una sola.

# COMMAND ----------

def analizar_portafolio(
    tickers: list[str],
    pesos: np.ndarray | None = None,
    covarianza: np.ndarray | None = None,
) -> dict:
    """Descompone el riesgo de un portafolio."""
    cov_total = cov_encogida if covarianza is None else covarianza
    indices = [matriz.columns.get_loc(t) for t in tickers]
    sigma = cov_total[np.ix_(indices, indices)]

    w = np.full(len(tickers), 1 / len(tickers)) if pesos is None else np.asarray(pesos, float)
    w = w / w.sum()

    varianza = float(w @ sigma @ w)
    vol = float(np.sqrt(varianza))

    vol_individual = np.sqrt(np.diag(sigma))
    contribucion_marginal = (sigma @ w) / vol
    contribucion = w * contribucion_marginal

    return {
        "tickers": tickers,
        "pesos": w,
        "vol_portafolio": vol,
        "vol_individual": vol_individual,
        "vol_promedio_ponderada": float(w @ vol_individual),
        "contribucion": contribucion,
        "pct_contribucion": contribucion / vol,
        "ratio_diversificacion": float((w @ vol_individual) / vol),
    }


def reporte(resultado: dict, titulo: str) -> None:
    print(f"═══ {titulo} ═══")
    print(f"  Volatilidad anual del portafolio  : {100 * resultado['vol_portafolio']:6.2f}%")
    print(f"  Promedio ponderado individual     : {100 * resultado['vol_promedio_ponderada']:6.2f}%")
    print(f"  Riesgo cancelado por correlación  : "
          f"{100 * (resultado['vol_promedio_ponderada'] - resultado['vol_portafolio']):6.2f} pp")
    print(f"  Ratio de diversificación          : {resultado['ratio_diversificacion']:6.2f}"
          f"   (máximo teórico {np.sqrt(len(resultado['tickers'])):.2f})")
    print()
    orden = np.argsort(-resultado["pct_contribucion"])
    print(f"  {'ticker':8} {'sector':24} {'peso':>7} {'vol':>7} {'% riesgo':>10} {'riesgo/peso':>12}")
    print("  " + "-" * 72)
    for i in orden:
        t = resultado["tickers"][i]
        sector = str(metadatos.loc[t, "sector"]) if t in metadatos.index else "?"
        razon = resultado["pct_contribucion"][i] / resultado["pesos"][i]
        print(f"  {t:8} {sector[:22]:24} {100 * resultado['pesos'][i]:6.1f}% "
              f"{100 * resultado['vol_individual'][i]:6.1f}% "
              f"{100 * resultado['pct_contribucion'][i]:9.1f}% {razon:11.2f}x")
    print()
    print(f"  Suma de contribuciones: {100 * resultado['pct_contribucion'].sum():.1f}% "
          "(debe dar 100%: el reparto es exacto)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 1 — El portafolio que *parece* diversificado
# MAGIC
# MAGIC Diez acciones, todas grandes y conocidas, en partes iguales. Cualquiera diría
# MAGIC que está diversificado: son diez empresas distintas.

# COMMAND ----------

CANDIDATOS_CONCENTRADO = ["JPM", "BAC", "WFC", "C", "GS", "MS", "PNC", "USB", "TFC", "RF"]
concentrado = [t for t in CANDIDATOS_CONCENTRADO if t in matriz.columns]

reporte(analizar_portafolio(concentrado), "DIEZ BANCOS, PESOS IGUALES")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Caso 2 — El mismo número de acciones, repartidas entre sectores

# COMMAND ----------

CANDIDATOS_DIVERSO = ["JPM", "AAPL", "JNJ", "XOM", "PG", "NEE", "AMT", "LIN", "UNP", "VZ"]
diverso = [t for t in CANDIDATOS_DIVERSO if t in matriz.columns]

reporte(analizar_portafolio(diverso), "DIEZ SECTORES DISTINTOS, PESOS IGUALES")

# COMMAND ----------

# MAGIC %md
# MAGIC ### La comparación

# COMMAND ----------

r_conc = analizar_portafolio(concentrado)
r_div = analizar_portafolio(diverso)

print(f"{'':34} {'10 bancos':>14} {'10 sectores':>14}")
print("-" * 64)
print(f"{'volatilidad del portafolio':34} {100 * r_conc['vol_portafolio']:13.2f}% "
      f"{100 * r_div['vol_portafolio']:13.2f}%")
print(f"{'promedio individual':34} {100 * r_conc['vol_promedio_ponderada']:13.2f}% "
      f"{100 * r_div['vol_promedio_ponderada']:13.2f}%")
print(f"{'ratio de diversificación':34} {r_conc['ratio_diversificacion']:14.2f} "
      f"{r_div['ratio_diversificacion']:14.2f}")
print()
print("Mismo número de posiciones, mismo peso por posición. La diferencia")
print("completa está en las correlaciones.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ¿Sirve de verdad el shrinkage?
# MAGIC
# MAGIC Todo lo anterior asume que la matriz encogida es mejor. Eso hay que
# MAGIC **demostrarlo**, no suponerlo, y la única prueba válida es **fuera de
# MAGIC muestra**.
# MAGIC
# MAGIC El experimento: partir la historia en dos. Con la primera mitad se estima la
# MAGIC covarianza y se construye el portafolio de **mínima varianza** — el que el
# MAGIC modelo cree que será el más estable. Con la segunda mitad, que el modelo
# MAGIC nunca vio, se mide la volatilidad que **realmente** tuvo.
# MAGIC
# MAGIC Si el shrinkage sirve, su portafolio debe resultar más estable en el futuro
# MAGIC que el construido con la matriz muestral.
# MAGIC
# MAGIC Se usa un subconjunto de 150 activos para que la matriz muestral sea al menos
# MAGIC invertible: con 1,211 activos y la mitad de la historia, simplemente no lo es,
# MAGIC y el experimento no tendría contra qué comparar.

# COMMAND ----------

rng = np.random.default_rng(42)
seleccion = sorted(rng.choice(matriz.columns, size=150, replace=False))
sub = matriz[seleccion].to_numpy(dtype=np.float64)

corte = int(len(sub) * 0.7)
entrena, prueba = sub[:corte], sub[corte:]

print(f"  entrenamiento: {entrena.shape[0]:,} días x {entrena.shape[1]} activos")
print(f"  prueba:        {prueba.shape[0]:,} días (nunca vistos)")
print()


def minima_varianza(sigma: np.ndarray) -> np.ndarray:
    """Pesos del portafolio de minima varianza: w = Σ⁻¹1 / (1ᵀΣ⁻¹1)."""
    unos = np.ones(sigma.shape[0])
    w = np.linalg.solve(sigma, unos)
    return w / w.sum()


cov_m = np.cov(entrena, rowvar=False)
cov_s, delta_sub, _ = shrinkage_ledoit_wolf(entrena)

w_muestral = minima_varianza(cov_m)
w_encogida = minima_varianza(cov_s)
w_iguales = np.full(len(seleccion), 1 / len(seleccion))

print(f"  δ sobre este subconjunto: {delta_sub:.4f}")
print()
print(f"{'estrategia':24} {'vol REAL':>12} {'vol prometida':>16} {'posición máx':>14} {'en corto':>10}")
print("-" * 80)

for nombre, w, cov_usada in [
    ("mínima var. muestral", w_muestral, cov_m),
    ("mínima var. encogida", w_encogida, cov_s),
    ("pesos iguales", w_iguales, None),
]:
    vol_real = float(np.std(prueba @ w) * np.sqrt(DIAS_HABILES))
    if cov_usada is None:
        prometida = float(np.sqrt(w @ np.cov(entrena, rowvar=False) @ w) * np.sqrt(DIAS_HABILES))
    else:
        prometida = float(np.sqrt(w @ cov_usada @ w) * np.sqrt(DIAS_HABILES))
    print(f"{nombre:24} {100 * vol_real:11.2f}% {100 * prometida:15.2f}% "
          f"{100 * np.abs(w).max():13.1f}% {100 * w[w < 0].sum():9.1f}%")

print()
print("La columna 'vol prometida' es lo que el modelo creía que iba a pasar; la")
print("'vol REAL' es lo que pasó. La brecha entre ambas mide cuánto se engañó el")
print("modelo consigo mismo, y es ahí donde la matriz muestral falla más feo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Guardar los resultados en el lake

# COMMAND ----------

tabla_cov = pd.DataFrame({
    "metrica": [
        "activos", "observaciones", "razon_T_N",
        "condicion_muestral", "condicion_encogida",
        "delta_shrinkage", "correlacion_media",
    ],
    "valor": [
        float(N), float(T), T / N,
        float(condicion), float(condicion_encogida),
        float(delta),
        float(np.mean((np.cov(X, rowvar=False) /
                       np.outer(np.sqrt(np.diag(np.cov(X, rowvar=False))),
                                np.sqrt(np.diag(np.cov(X, rowvar=False)))))[
            np.triu_indices(N, k=1)])),
    ],
})

(
    spark.createDataFrame(tabla_cov)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.covariance_diagnostics")
)

riesgo = pd.DataFrame({
    "portafolio": ["10_bancos"] * len(concentrado) + ["10_sectores"] * len(diverso),
    "ticker": concentrado + diverso,
    "peso": list(r_conc["pesos"]) + list(r_div["pesos"]),
    "vol_individual": list(r_conc["vol_individual"]) + list(r_div["vol_individual"]),
    "pct_riesgo": list(r_conc["pct_contribucion"]) + list(r_div["pct_contribucion"]),
})

(
    spark.createDataFrame(riesgo)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD}.portfolio_risk")
)

for t in ["covariance_diagnostics", "portfolio_risk"]:
    print(f"  {t:26} {spark.table(f'{GOLD}.{t}').count():>6,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qué sigue
# MAGIC
# MAGIC `06_regimenes_de_crisis` responde la otra mitad: este motor calcula el riesgo
# MAGIC **promedio de once años**, pero la correlación no es estable. En las crisis
# MAGIC se dispara, y con ella el riesgo de cualquier portafolio.
