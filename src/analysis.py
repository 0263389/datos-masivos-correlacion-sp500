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
