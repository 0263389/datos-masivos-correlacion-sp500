"""Motor de riesgo de portafolio.

Estimacion de covarianza y descomposicion del riesgo. Es la capa analitica
que esta en el centro de plataformas como Aladdin.

**Este modulo no importa pyspark ni src.config a proposito.** Solo depende de
numpy y pandas, por tres razones:

* Corre igual en una laptop y en Databricks, sin cambiar una linea.
* Se puede probar sin levantar Spark, que es lo que permite tener tests.
* Importar ``config`` arrastraria la creacion de carpetas y la lectura del
  ``.env``, que no tienen sentido dentro de un workspace de Databricks.

La frontera del proyecto esta justo aqui: Spark reduce millones de filas a una
matriz de pocas decenas de megabytes, y de ahi en adelante el trabajo es
algebra lineal densa, que es lo que numpy hace bien y Spark hace mal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_HABILES = 252  # dias de mercado en un anio, para anualizar


# ==========================================================================
# 1. Estimacion de la covarianza
# ==========================================================================
def sample_covariance(X: np.ndarray, annualize: bool = True) -> np.ndarray:
    """Covarianza muestral insesgada.

    Parameters
    ----------
    X
        Matriz de rendimientos, con las fechas en las filas y los activos en
        las columnas.
    """
    cov = np.cov(X, rowvar=False)
    return cov * DIAS_HABILES if annualize else cov


def condition_number(cov: np.ndarray) -> float:
    """Cociente entre el mayor y el menor valor propio.

    Mide cuanto amplifica los errores al invertir la matriz, y para optimizar
    un portafolio hay que invertirla. Por encima de 1e6 el resultado de la
    inversion es ruido amplificado, no senial.
    """
    valores = np.linalg.eigvalsh(cov)
    return float(valores[-1] / max(valores[0], 1e-20))


def ledoit_wolf_shrinkage(
    X: np.ndarray, annualize: bool = True
) -> tuple[np.ndarray, float, np.ndarray]:
    """Covarianza encogida hacia el objetivo de correlacion constante.

    Implementa Ledoit & Wolf (2003), *Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection*.

    **Por que hace falta.** Con N activos hay N(N+1)/2 parametros que estimar
    y solo T observaciones. Con 1,200 activos y 2,900 dias son 720,000
    incognitas contra 2,900 datos: la matriz muestral queda tan mal
    condicionada que invertirla amplifica el ruido.

    **Como funciona.** Se promedia la matriz muestral con una matriz
    estructurada y estable::

        Sigma = delta * F + (1 - delta) * S

    El objetivo ``F`` conserva la volatilidad real de cada activo pero asume
    que todos los pares tienen la misma correlacion, el promedio observado.
    Es rigido, pero estima un solo parametro en lugar de cientos de miles, asi
    que casi no tiene ruido.

    Lo valioso del metodo es que ``delta`` **no se elige a mano**: se deriva en
    forma cerrada el valor que minimiza el error cuadratico esperado. Depende
    de cuanto ruido tiene la muestral (``pi``) frente a cuanto se equivoca el
    objetivo (``gamma``).

    Returns
    -------
    (covarianza encogida, delta optimo, matriz objetivo F)
    """
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    if T < 2 or N < 2:
        raise ValueError("Se necesitan al menos 2 observaciones y 2 activos.")

    Y = X - X.mean(axis=0)
    # Covarianza sesgada (divide entre T, no entre T-1): es la que usa el
    # paper, y mezclar convenciones aqui descuadra el calculo de delta.
    S = (Y.T @ Y) / T

    desv = np.sqrt(np.diag(S))
    if np.any(desv <= 0):
        raise ValueError("Hay activos con varianza cero; quitalos antes.")

    corr = S / np.outer(desv, desv)
    triangulo = np.triu_indices(N, k=1)
    r_media = float(np.mean(corr[triangulo]))

    # F: misma volatilidad individual, correlacion comun para todos los pares.
    F = r_media * np.outer(desv, desv)
    np.fill_diagonal(F, np.diag(S))

    # pi: cuanto ruido tiene cada entrada de la covarianza muestral.
    Y2 = Y**2
    pi_matriz = (Y2.T @ Y2) / T - S**2
    pi = float(pi_matriz.sum())

    # rho: la parte de ese ruido que el objetivo tambien comparte, y que por lo
    # tanto no se gana nada encogiendo. Los terminos cruzados usan theta, que
    # mide como covaria el error de la varianza de un activo con el error de su
    # covarianza con otro.
    Y3 = Y**3
    theta = (Y3.T @ Y) / T - np.diag(S)[:, None] * S

    razon = np.outer(desv, 1.0 / desv)
    cruzados = (r_media / 2.0) * (razon * theta.T + razon.T * theta)
    np.fill_diagonal(cruzados, 0.0)
    rho = float(np.diag(pi_matriz).sum()) + float(cruzados.sum())

    # gamma: que tan lejos esta el objetivo de la muestral.
    gamma = float(((F - S) ** 2).sum())

    delta = max(0.0, min(1.0, ((pi - rho) / gamma) / T)) if gamma > 0 else 0.0

    cov = delta * F + (1 - delta) * S
    if annualize:
        cov, F = cov * DIAS_HABILES, F * DIAS_HABILES
    return cov, delta, F


# ==========================================================================
# 2. Construccion de portafolios
# ==========================================================================
def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Pesos del portafolio de minima varianza: ``w = inv(Sigma) 1 / 1' inv(Sigma) 1``.

    Sin restriccion de posiciones largas, que es lo que lo vuelve una prueba
    exigente de la calidad de la covarianza: con una matriz mal condicionada,
    la optimizacion produce posiciones enormes y en corto que no sobreviven
    fuera de muestra.

    Se usa ``solve`` y no ``inv`` porque resolver el sistema es mas estable
    numericamente que invertir y multiplicar.
    """
    unos = np.ones(cov.shape[0])
    w = np.linalg.solve(cov, unos)
    return w / w.sum()


def equal_weights(n: int) -> np.ndarray:
    """Pesos iguales. Es el punto de comparacion honesto: cualquier modelo de
    covarianza deberia superarlo, y con frecuencia no lo logra."""
    return np.full(n, 1.0 / n)


# ==========================================================================
# 3. Descomposicion del riesgo
# ==========================================================================
def risk_decomposition(
    cov: np.ndarray,
    weights: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Descompone el riesgo de un portafolio en la contribucion de cada posicion.

    Cuatro cantidades:

    ===========================  ============================  ================
    Medida                       Formula                       Responde
    ===========================  ============================  ================
    Volatilidad                  ``sqrt(w' Sigma w)``          cuanto se mueve
    Contribucion al riesgo       ``w_i (Sigma w)_i / sigma``   cuanto aporta
    Ratio de diversificacion     ``(w' sigma_i) / sigma_p``    cuanto cancelo
    Volatilidad standalone       ``sqrt(Sigma_ii)``            si estuviera sola
    ===========================  ============================  ================

    La contribucion al riesgo es la pieza no obvia: **no es proporcional al
    dinero invertido**. Una posicion del 5% puede aportar el 30% del riesgo si
    es volatil y esta correlacionada con el resto. Las contribuciones suman
    exactamente la volatilidad del portafolio, asi que el reparto es una
    identidad y no una aproximacion -- ``check_sum`` lo verifica.

    El ratio de diversificacion tiene lectura directa: cuantas posiciones
    verdaderamente independientes hay. Un ratio de 1.0 significa que todas las
    posiciones se comportan como una sola; el maximo alcanzable es
    ``sqrt(n)``, y solo con activos sin correlacion alguna.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]

    w = equal_weights(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape[0] != n:
        raise ValueError(f"Los pesos ({w.shape[0]}) no coinciden con la matriz ({n}).")
    w = w / w.sum()

    varianza = float(w @ cov @ w)
    if varianza <= 0:
        raise ValueError("La varianza del portafolio salio no positiva.")
    vol = float(np.sqrt(varianza))

    vol_individual = np.sqrt(np.diag(cov))
    contribucion_marginal = (cov @ w) / vol
    contribucion = w * contribucion_marginal
    vol_ponderada = float(w @ vol_individual)

    return {
        "labels": labels if labels is not None else [f"a{i}" for i in range(n)],
        "weights": w,
        "portfolio_vol": vol,
        "standalone_vol": vol_individual,
        "weighted_avg_vol": vol_ponderada,
        "marginal_contribution": contribucion_marginal,
        "risk_contribution": contribucion,
        "pct_risk": contribucion / vol,
        "diversification_ratio": vol_ponderada / vol,
        "max_diversification_ratio": float(np.sqrt(n)),
        # Debe ser practicamente cero: las contribuciones suman la volatilidad.
        "check_sum": float(abs(contribucion.sum() - vol)),
    }


def risk_report(result: dict, title: str = "PORTAFOLIO", sectors: dict | None = None) -> None:
    """Imprime la descomposicion como texto, ordenada por contribucion."""
    print(f"═══ {title} ═══")
    print(f"  Volatilidad anual del portafolio  : {100 * result['portfolio_vol']:6.2f}%")
    print(f"  Promedio ponderado individual     : {100 * result['weighted_avg_vol']:6.2f}%")
    print(f"  Riesgo cancelado por correlación  : "
          f"{100 * (result['weighted_avg_vol'] - result['portfolio_vol']):6.2f} pp")
    print(f"  Ratio de diversificación          : {result['diversification_ratio']:6.2f}"
          f"   (máximo teórico {result['max_diversification_ratio']:.2f})")
    print()
    print(f"  {'activo':9} {'sector':22} {'peso':>7} {'vol':>7} {'% riesgo':>10} {'riesgo/peso':>12}")
    print("  " + "-" * 71)
    for i in np.argsort(-result["pct_risk"]):
        etiqueta = result["labels"][i]
        sector = (sectors or {}).get(etiqueta, "")
        razon = result["pct_risk"][i] / result["weights"][i]
        print(f"  {etiqueta:9} {str(sector)[:20]:22} {100 * result['weights'][i]:6.1f}% "
              f"{100 * result['standalone_vol'][i]:6.1f}% "
              f"{100 * result['pct_risk'][i]:9.1f}% {razon:11.2f}x")
    print()
    print(f"  Suma de contribuciones: {100 * result['pct_risk'].sum():.2f}%  "
          f"(error {result['check_sum']:.2e})")


# ==========================================================================
# 4. Regimenes de mercado
# ==========================================================================
def classify_regimes(
    monthly_returns: pd.Series, labels: tuple[str, str, str] = (
        "meses malos", "meses normales", "meses buenos"
    )
) -> pd.Series:
    """Parte los meses en tercios por rendimiento.

    Se usan tercios y no un umbral fijo porque reparte los meses en grupos de
    tamanio comparable, lo que hace que los promedios por grupo sean
    igualmente confiables. Un umbral del tipo "mes malo es menos de -2%" daria
    grupos muy desiguales y un promedio de crisis calculado con cuatro
    observaciones.
    """
    bajo, alto = monthly_returns.quantile([1 / 3, 2 / 3])
    return monthly_returns.map(
        lambda r: labels[0] if r <= bajo else (labels[2] if r >= alto else labels[1])
    )


def regime_comparison(
    returns: pd.DataFrame,
    regimes: pd.Series,
    tickers: list[str],
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compara el riesgo del mismo portafolio bajo distintos regimenes.

    Es la prueba que desmonta el supuesto del motor de riesgo: si el ratio de
    diversificacion baja en los meses malos, las correlaciones subieron, y las
    mismas posiciones protegen menos justo cuando hace falta que protejan.

    Parameters
    ----------
    returns
        Rendimientos diarios, indexados por fecha, con un activo por columna.
    regimes
        Regimen de cada mes, indexado por periodo mensual.
    """
    periodos = returns.index.to_period("M")
    filas = []

    for regimen in ["todo el periodo", *dict.fromkeys(regimes.tolist())]:
        if regimen == "todo el periodo":
            mascara = np.ones(len(returns), dtype=bool)
        else:
            meses = set(regimes[regimes == regimen].index)
            mascara = periodos.isin(meses)

        sub = returns.loc[mascara, tickers].to_numpy(dtype=np.float64)
        if len(sub) < len(tickers) + 2:
            continue

        resultado = risk_decomposition(sample_covariance(sub), weights, tickers)
        filas.append({
            "regimen": regimen,
            "dias": len(sub),
            "vol_portafolio": resultado["portfolio_vol"],
            "vol_individual": resultado["weighted_avg_vol"],
            "ratio_diversificacion": resultado["diversification_ratio"],
        })

    return pd.DataFrame(filas)


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Proyeccion euclidiana al simplex: pesos no negativos que suman 1.

    Algoritmo de Duchi et al. (2008). Encuentra el punto del simplex mas
    cercano a ``v``, que es lo que permite forzar la restriccion en cada paso
    del descenso sin deformar la direccion de busqueda mas de lo necesario.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    acumulado = np.cumsum(u)
    indices = np.arange(1, n + 1)
    validos = u + (1 - acumulado) / indices > 0
    rho = indices[validos][-1]
    theta = (1 - acumulado[rho - 1]) / rho
    return np.maximum(v + theta, 0.0)


def min_variance_weights_long_only(
    cov: np.ndarray, max_iter: int = 5000, tol: float = 1e-11
) -> np.ndarray:
    """Minima varianza con la restriccion de no vender en corto.

    **Por que hace falta.** La version sin restricciones produce posiciones en
    corto enormes -en este proyecto llego a -155% del capital- porque explota
    los errores de estimacion de la covarianza: si el modelo cree por ruido que
    dos activos se cancelan, la optimizacion apalanca esa creencia sin limite.
    El resultado es matematicamente optimo e imposible de implementar.

    Exigir pesos no negativos actua como una regularizacion implicita: acota lo
    que la optimizacion puede apostar a una correlacion estimada, asi que el
    portafolio resultante sobrevive mucho mejor fuera de muestra.

    Se resuelve con descenso de gradiente proyectado. El gradiente de
    ``w' Sigma w`` es ``2 Sigma w``, y tras cada paso se proyecta de vuelta al
    simplex. El paso se fija con el mayor valor propio, que acota la curvatura
    del problema y garantiza que el descenso no se pase.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)

    paso = 1.0 / (2.0 * float(np.linalg.eigvalsh(cov)[-1]))

    for _ in range(max_iter):
        nuevo = _project_to_simplex(w - paso * (2.0 * cov @ w))
        if np.max(np.abs(nuevo - w)) < tol:
            w = nuevo
            break
        w = nuevo

    return w
