"""Transformacion de precios a rendimientos.

Este es el modulo conceptualmente mas importante del proyecto.

**Por que no se correlacionan precios.** Una serie de precios no es
estacionaria: su media y su varianza cambian con el tiempo porque tiene
tendencia. Si correlacionas el precio de dos empresas que subieron en la
ultima decada, obtienes ~0.95 y eso no significa que se muevan juntas:
significa que las dos subieron. Es correlacion espuria. Podrias correlacionar
el precio de Apple con la poblacion de Mexico y obtener algo parecido.

**La solucion.** Trabajar con rendimientos logaritmicos:

    r_t = ln(P_t / P_{t-1})

Ventajas: son aproximadamente estacionarios, son aditivos en el tiempo (el
rendimiento de dos dias es la suma de los dos rendimientos diarios) y tratan
de forma simetrica las subidas y las bajadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def filter_by_history(
    prices: pd.DataFrame, min_observations: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Descarta tickers sin suficiente historia para estimar correlaciones.

    Una empresa que salio a bolsa hace seis meses no tiene datos para una
    correlacion confiable. Incluirla no solo agrega ruido: obliga a estimar
    cada par con un numero distinto de observaciones.

    Returns
    -------
    (precios filtrados, lista de tickers descartados)
    """
    min_observations = min_observations or config.MIN_OBSERVATIONS

    counts = prices.notna().sum()
    keep = counts[counts >= min_observations].index
    dropped = sorted(set(prices.columns) - set(keep))
    return prices[keep], dropped


def build_balanced_panel(
    prices: pd.DataFrame, min_observations: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Construye un panel rectangular sin huecos.

    Dos pasos, en este orden (el orden importa):

    1. Descartar tickers con poca historia.
    2. Descartar las fechas que aun tengan algun faltante.

    Si se hiciera al reves, un solo ticker reciente borraria anios enteros de
    historia para todos los demas.

    Deliberadamente **no** se rellena hacia adelante (``ffill``): eso inventaria
    dias con rendimiento de 0% que nunca ocurrieron y sesgaria las
    correlaciones hacia abajo.

    Returns
    -------
    (panel balanceado, diccionario con el registro de lo que se descarto)
    """
    filtered, dropped_tickers = filter_by_history(prices, min_observations)

    rows_before = len(filtered)
    balanced = filtered.dropna(axis=0, how="any")

    log = {
        "tickers_iniciales": prices.shape[1],
        "tickers_descartados": len(dropped_tickers),
        "tickers_finales": balanced.shape[1],
        "lista_descartados": dropped_tickers,
        "filas_iniciales": rows_before,
        "filas_descartadas": rows_before - len(balanced),
        "filas_finales": len(balanced),
    }
    return balanced, log


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convierte precios ajustados a rendimientos logaritmicos diarios.

    La primera fila queda sin definir (no hay dia previo con que comparar) y
    se elimina.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.iloc[1:]


def annualize_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Anualiza la desviacion estandar de rendimientos diarios.

    Se escala por la raiz del numero de periodos porque la **varianza** crece
    de forma lineal con el tiempo, no la desviacion estandar. 252 es el numero
    aproximado de dias habiles de mercado en un anio.
    """
    return returns.std() * np.sqrt(periods_per_year)
