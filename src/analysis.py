"""Analisis de correlacion y agregacion por sector.

Responde la pregunta del Entregable 1:

    Las acciones del mismo sector, se mueven mas parecido entre si que con
    acciones de otros sectores?

Estrategia: convertir la matriz de correlacion en una tabla de **pares
unicos**, etiquetar cada par como intra-sector o inter-sector, y comparar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Matriz de correlacion entre rendimientos.

    ``pearson`` mide relacion lineal; ``spearman`` mide relacion monotona sobre
    los rangos y es mas robusta a valores extremos. Comparar ambas es una
    prueba util: si difieren mucho, el resultado depende de unos pocos dias
    atipicos.
    """
    return returns.corr(method=method)


def pair_table(
    correlation: pd.DataFrame, constituents: pd.DataFrame
) -> pd.DataFrame:
    """Convierte la matriz de correlacion en una tabla larga de pares unicos.

    Se usa solo el **triangulo superior** de la matriz. Tomarla completa
    contaria cada par dos veces (A-B y B-A) e incluiria la diagonal, que
    siempre vale 1.0 e inflaria cualquier promedio.

    Returns
    -------
    DataFrame con: ticker_a, ticker_b, corr, sector_a, sector_b, same_sector.
    """
    tickers = correlation.columns.to_numpy()
    values = correlation.to_numpy()

    # k=1 excluye la diagonal principal.
    i, j = np.triu_indices(len(tickers), k=1)

    pairs = pd.DataFrame(
        {
            "ticker_a": tickers[i],
            "ticker_b": tickers[j],
            "corr": values[i, j],
        }
    )

    sector_of = constituents.set_index("ticker")["sector"]
    company_of = constituents.set_index("ticker")["company"]

    pairs["sector_a"] = pairs["ticker_a"].map(sector_of)
    pairs["sector_b"] = pairs["ticker_b"].map(sector_of)
    pairs["company_a"] = pairs["ticker_a"].map(company_of)
    pairs["company_b"] = pairs["ticker_b"].map(company_of)
    pairs["same_sector"] = pairs["sector_a"] == pairs["sector_b"]

    return pairs


def sector_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compara la correlacion promedio intra-sector contra inter-sector.

    Esta es la respuesta directa a la pregunta del entregable.
    """
    summary = (
        pairs.groupby("same_sector")["corr"]
        .agg(n_pares="size", media="mean", mediana="median", desv_est="std")
        .rename(index={True: "Mismo sector", False: "Distinto sector"})
    )
    summary.index.name = "grupo"
    return summary.reindex(["Mismo sector", "Distinto sector"])


def sector_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Matriz 11x11 de correlacion promedio entre cada par de sectores.

    La diagonal es la cohesion interna de cada sector: que tan parecido se
    mueven las empresas de esa industria entre si.
    """
    # Cada par aparece una sola vez, asi que se agrega en ambas direcciones
    # para que la matriz resultante sea simetrica.
    forward = pairs[["sector_a", "sector_b", "corr"]]
    backward = pairs[["sector_b", "sector_a", "corr"]]
    backward.columns = ["sector_a", "sector_b", "corr"]

    both = pd.concat([forward, backward], ignore_index=True)
    matrix = both.pivot_table(
        index="sector_a", columns="sector_b", values="corr", aggfunc="mean"
    )
    matrix.index.name = "sector"
    matrix.columns.name = "sector"
    return matrix


def sector_cohesion(pairs: pd.DataFrame) -> pd.DataFrame:
    """Ordena los sectores por que tan cohesionados estan internamente."""
    intra = pairs[pairs["same_sector"]]
    cohesion = (
        intra.groupby("sector_a")["corr"]
        .agg(n_pares="size", correlacion_media="mean")
        .sort_values("correlacion_media", ascending=False)
    )
    cohesion.index.name = "sector"
    return cohesion


def top_pairs(pairs: pd.DataFrame, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Los ``n`` pares mas (o menos) correlacionados del universo.

    Prueba de sentido comun: los pares mas correlacionados deberian ser
    empresas que cualquiera reconoceria como parecidas. Si no lo son, hay algo
    mal en el pipeline.
    """
    columns = ["ticker_a", "company_a", "ticker_b", "company_b", "sector_a", "sector_b", "corr"]
    return (
        pairs.sort_values("corr", ascending=ascending)
        .head(n)[columns]
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Robustez: Pearson contra Spearman
# --------------------------------------------------------------------------
def _unique_pair_values(correlation: pd.DataFrame) -> np.ndarray:
    """Valores del triangulo superior de la matriz, sin la diagonal.

    Mismo criterio que ``pair_table``: cada par cuenta una sola vez y la
    diagonal queda fuera porque vale 1.0 con cualquier metodo. Incluirla no
    cambia el signo de nada, pero encoge cualquier promedio por un factor de
    n/(n-1) al meter ceros que no son un resultado, son la definicion.
    """
    i, j = np.triu_indices(len(correlation), k=1)
    return correlation.to_numpy()[i, j]


def method_difference(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.Series:
    """Compara ambos metodos par por par, sobre los pares unicos.

    Reportar solo el promedio de las diferencias esconde el caso interesante.
    El promedio puede salir chico y aun asi existir pares donde los dos
    metodos discrepan fuerte: son justamente los pares cuya correlacion la
    sostienen unos pocos dias extremos. Por eso se devuelven tambien mediana,
    percentil 95 y maximo, ademas de la correlacion entre los dos metodos.
    """
    if not pearson.columns.equals(spearman.columns):
        raise ValueError(
            "Ambas matrices deben tener los mismos tickers en el mismo orden."
        )

    a = _unique_pair_values(pearson)
    b = _unique_pair_values(spearman)
    diff = np.abs(a - b)

    return pd.Series(
        {
            "n_pares": float(len(diff)),
            "pearson_media": a.mean(),
            "spearman_media": b.mean(),
            "dif_media": diff.mean(),
            "dif_mediana": float(np.median(diff)),
            "dif_p95": float(np.percentile(diff, 95)),
            "dif_maxima": diff.max(),
            # Correlacion entre ambos metodos tratando cada par como una
            # observacion: mide si ordenan los pares igual, aunque los niveles
            # difieran. Es la version exigente de la prueba, porque un promedio
            # de diferencias chico tambien se puede lograr con mucho ruido que
            # se cancela.
            "corr_entre_metodos": float(np.corrcoef(a, b)[0, 1]),
        },
        name="pearson vs spearman",
    )


def compare_methods(
    returns: pd.DataFrame, constituents: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Repite el analisis sectorial completo con Pearson y con Spearman.

    Pearson mide relacion lineal y le da todo el peso a los dias de
    movimientos grandes: un dia de -18% pesa mucho mas que uno de -4%.
    Spearman trabaja sobre rangos, donde esos dos dias solo son "el peor" y
    "el cuarto peor". Correr ambas es una prueba de robustez.

    La pregunta que importa no es si las dos matrices son identicas -nunca lo
    son- sino si la **conclusion del entregable** sobrevive al cambio de
    metodo: si sigue habiendo brecha entre los pares del mismo sector y los
    de sectores distintos. Por eso no se comparan matrices sueltas sino el
    resultado final que sale de cada una.

    Returns
    -------
    (tabla comparativa por grupo, estadisticas de la diferencia por par)
    """
    matrices = {
        method: correlation_matrix(returns, method=method)
        for method in ("pearson", "spearman")
    }
    medias = {
        method: sector_comparison(pair_table(correlation, constituents))["media"]
        for method, correlation in matrices.items()
    }

    table = pd.DataFrame(medias)
    # La brecha es el resultado del entregable. Que aguante el cambio de
    # metodo es lo que lo vuelve creible.
    table.loc["Brecha"] = table.loc["Mismo sector"] - table.loc["Distinto sector"]
    table["diferencia"] = table["spearman"] - table["pearson"]
    table.index.name = "grupo"

    return table, method_difference(matrices["pearson"], matrices["spearman"])


def distance_matrix(correlation: pd.DataFrame) -> pd.DataFrame:
    """Convierte la matriz de correlacion en una matriz de distancia.

        d(i, j) = sqrt(2 * (1 - rho(i, j)))

    Metrica de Mantegna (1999), "Hierarchical structure in financial
    markets", European Physical Journal B 11, 193-197.

    La correlacion no se puede usar como distancia tal cual: rho = 1 significa
    "se mueven identico" y una distancia tiene que valer 0 justo ahi. Esta
    transformacion invierte la escala y ademas cumple la desigualdad del
    triangulo, que es la propiedad que permite meterla a un clustering
    jerarquico o a un arbol de expansion minima.

    Rango: 0 cuando rho = 1, sqrt(2) ~ 1.414 cuando rho = 0, y 2 cuando
    rho = -1.
    """
    values = 2.0 * (1.0 - correlation.to_numpy())

    # Con una matriz salida de ``corr()`` la diagonal es exactamente 1.0 y esto
    # da 0.0 limpio. Una matriz de otra fuente puede traer 1.0000000000000002 y
    # producir la raiz de un numero negativo; el clip lo vuelve inofensivo sin
    # alterar ningun valor legitimo.
    return pd.DataFrame(
        np.sqrt(np.clip(values, 0.0, None)),
        index=correlation.index,
        columns=correlation.columns,
    )


def rolling_correlation(
    returns: pd.DataFrame, ticker_a: str, ticker_b: str, window: int = 252
) -> pd.Series:
    """Correlacion entre dos acciones a lo largo del tiempo.

    Una matriz de correlacion resume once anios en un solo numero, y al
    hacerlo esconde lo mas interesante: la correlacion no es constante. Sube
    en las crisis, es decir, la diversificacion se debilita justo cuando
    tendria que estar protegiendo.

    **Como leer la serie.** La ventana es retrospectiva y el valor se fecha al
    final de ella: el dato del 17-mar-2020 resume el anio que **termina** ese
    dia, no ese dia. Dos consecuencias al interpretar la grafica:

    - El pico no marca el inicio de la crisis, marca el momento en que la
      crisis ya entro completa a la ventana.
    - La correlacion se queda alta cerca de doce meses despues del evento
      porque la ventana lo sigue conteniendo, no porque el mercado siga
      estresado. Medido con JPM-BAC: baja de 0.953 a 0.911 entre enero y junio
      de 2021, justo cuando marzo de 2020 termina de salir de la ventana.

    ``window`` son 252 dias habiles por omision, un anio de mercado.
    """
    faltantes = [t for t in (ticker_a, ticker_b) if t not in returns.columns]
    if faltantes:
        raise KeyError(f"No estan en el panel de rendimientos: {faltantes}")

    pair = returns[ticker_a].rolling(window).corr(returns[ticker_b])
    pair.name = f"{ticker_a}-{ticker_b}"

    # Las primeras ``window - 1`` fechas no tienen ventana completa.
    return pair.dropna()
