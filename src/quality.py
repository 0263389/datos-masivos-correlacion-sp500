"""Diagnostico de calidad de los datos.

Responde a la seccion "Diagnostico inicial" del entregable: valores faltantes,
duplicados, inconsistencias y limitaciones.

La idea de fondo: **antes de analizar hay que auditar**. Una matriz de
correlacion no avisa cuando esta calculada sobre datos rotos; simplemente
devuelve numeros que parecen razonables.
"""

from __future__ import annotations

import pandas as pd

from src import config


def structure_summary(prices: pd.DataFrame) -> pd.Series:
    """Resumen global del panel de precios: forma, periodo, nulos, duplicados."""
    total_cells = prices.shape[0] * prices.shape[1]
    missing_cells = int(prices.isna().sum().sum())

    return pd.Series(
        {
            "filas (dias de mercado)": prices.shape[0],
            "columnas (tickers)": prices.shape[1],
            "observaciones totales": total_cells,
            "valores faltantes": missing_cells,
            "% faltantes": round(100 * missing_cells / total_cells, 2),
            "fechas duplicadas": int(prices.index.duplicated().sum()),
            "tickers duplicados": int(pd.Index(prices.columns).duplicated().sum()),
            "fecha inicial": prices.index.min().date(),
            "fecha final": prices.index.max().date(),
        }
    )


def per_ticker_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Diagnostico por ticker: cuanta historia tiene cada empresa.

    La columna clave es ``n_obs``. Un ticker con pocas observaciones no es un
    dato "sucio": es una empresa que salio a bolsa despues del inicio del
    periodo. La distincion importa porque el tratamiento es distinto.
    """
    report = pd.DataFrame(
        {
            "n_obs": prices.notna().sum(),
            "n_missing": prices.isna().sum(),
            "first_date": prices.apply(lambda col: col.first_valid_index()),
            "last_date": prices.apply(lambda col: col.last_valid_index()),
        }
    )
    report["pct_missing"] = (
        100 * report["n_missing"] / len(prices)
    ).round(2)
    report["suficiente_historia"] = report["n_obs"] >= config.MIN_OBSERVATIONS
    report.index.name = "ticker"
    return report.sort_values("n_obs")


def find_extreme_returns(
    returns: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Localiza rendimientos diarios sospechosamente grandes.

    Un movimiento de +/-50% en un dia en una empresa del S&P 500 es posible
    (paso en marzo de 2020 y en algunas fusiones), pero es mucho mas frecuente
    que sea un split mal ajustado. Hay que **verlos uno por uno**, no borrarlos
    automaticamente: borrar datos reales sesga el analisis tanto como
    conservar datos falsos.

    Returns
    -------
    DataFrame largo con columnas: date, ticker, log_return.
    """
    threshold = threshold if threshold is not None else config.EXTREME_RETURN_THRESHOLD

    # .dropna() es indispensable: a partir de pandas 3.0, .stack() ya no
    # elimina los nulos por defecto, asi que sin esto se devolveria el panel
    # completo en lugar de solo los casos extremos.
    extremes = (
        returns[returns.abs() > threshold]
        .stack()
        .dropna()
        .rename("log_return")
        .reset_index()
    )
    extremes.columns = ["date", "ticker", "log_return"]
    return extremes.sort_values("log_return", key=abs, ascending=False)


def flag_suspicious_tickers(
    returns: pd.DataFrame,
    threshold: float | None = None,
    max_events: int | None = None,
) -> pd.DataFrame:
    """Identifica tickers cuyos datos parecen corruptos, no extremos.

    El criterio distingue dos cosas que se ven iguales en la tabla de
    rendimientos extremos:

    * **Evento real.** Una empresa se desploma una vez y no se recupera al dia
      siguiente. Ejemplos encontrados en estos datos: PCG en enero de 2019
      (bancarrota de PG&E por los incendios de California), APA / OXY / TRGP el
      9 de marzo de 2020 (guerra de precios del petroleo y desplome de COVID).
      Estos datos son correctos y deben conservarse.

    * **Error de la fuente.** El precio salta a la mitad y regresa, varias
      veces, sin ninguna noticia detras. Es un split aplicado de forma
      inconsistente. Ejemplo encontrado: MNST (Monster Beverage), que oscila
      entre ~95 y ~47 dolares en dias sueltos de 2026.

    La regla que los separa es la **frecuencia**: una empresa real tiene a lo
    mucho uno o dos eventos de esta magnitud en once anios. Un ticker con siete
    no esta siendo volatil, esta mal medido.

    Returns
    -------
    DataFrame con: ticker, n_eventos_extremos, sospechoso.
    """
    threshold = threshold if threshold is not None else config.EXTREME_RETURN_THRESHOLD
    max_events = max_events if max_events is not None else config.MAX_EXTREME_EVENTS

    counts = (returns.abs() > threshold).sum()
    report = pd.DataFrame({"n_eventos_extremos": counts})
    report["sospechoso"] = report["n_eventos_extremos"] > max_events
    report.index.name = "ticker"
    return report[report["n_eventos_extremos"] > 0].sort_values(
        "n_eventos_extremos", ascending=False
    )


def calendar_gaps(prices: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Detecta huecos inusuales en el calendario de mercado.

    Los fines de semana y feriados producen huecos normales de 1 a 4 dias. Un
    hueco mayor sugiere datos faltantes en la fuente, no un feriado.
    """
    gaps = prices.index.to_series().diff().dt.days
    flagged = gaps[gaps > max_gap_days]
    return pd.DataFrame({"fecha": flagged.index, "dias_sin_datos": flagged.values})


def missing_by_year(prices: pd.DataFrame) -> pd.DataFrame:
    """Distribucion de los valores faltantes anio por anio.

    Contar los nulos totales no dice nada por si solo: 40,000 huecos pueden ser
    una fuente rota o pueden ser empresas que todavia no existian. Lo que
    distingue un caso del otro es **donde** cae el hueco dentro de la historia
    de cada ticker:

    * ``previos_al_listado`` - el hueco esta antes de la primera cotizacion de
      la empresa. No es un dato perdido: la accion aun no se negociaba. Se
      espera que se concentren en los anios iniciales del periodo.

    * ``posteriores_a_la_baja`` - el hueco esta despues de la ultima
      cotizacion. La empresa salio del indice, fue adquirida o cambio de
      ticker. Tampoco es un error de la fuente.

    * ``huecos_internos`` - la empresa ya cotizaba y volvio a cotizar despues,
      pero ese dia no hay precio. **Estos son los unicos preocupantes.** Si
      aparecen repartidos por todos los anios apuntan a una descarga
      incompleta; si se agrupan en fechas concretas, a dias que la fuente no
      entrego.

    La suma de las tres columnas es ``n_faltantes``, de modo que el desglose
    siempre cuadra con el total que reporta :func:`structure_summary`.

    Returns
    -------
    DataFrame indexado por anio con: dias_de_mercado, n_faltantes,
    pct_faltantes, previos_al_listado, posteriores_a_la_baja, huecos_internos.
    """
    missing = prices.isna()

    # Marca, para cada celda, si el ticker ya habia cotizado alguna vez
    # (cumsum hacia adelante) y si volveria a cotizar despues (cumsum hacia
    # atras). Una celda vacia entre ambos limites es un hueco real.
    ya_cotizaba = prices.notna().cumsum() > 0
    volvera_a_cotizar = prices.notna()[::-1].cumsum()[::-1] > 0

    previos = missing & ~ya_cotizaba
    posteriores = missing & ~volvera_a_cotizar
    internos = missing & ya_cotizaba & volvera_a_cotizar

    anio = prices.index.year
    report = pd.DataFrame(
        {
            "dias_de_mercado": missing.groupby(anio).size(),
            "n_faltantes": missing.sum(axis=1).groupby(anio).sum(),
            "previos_al_listado": previos.sum(axis=1).groupby(anio).sum(),
            "posteriores_a_la_baja": posteriores.sum(axis=1).groupby(anio).sum(),
            "huecos_internos": internos.sum(axis=1).groupby(anio).sum(),
        }
    )
    report["pct_faltantes"] = (
        100 * report["n_faltantes"] / (report["dias_de_mercado"] * prices.shape[1])
    ).round(2)
    report.index.name = "anio"

    return report[
        [
            "dias_de_mercado",
            "n_faltantes",
            "pct_faltantes",
            "previos_al_listado",
            "posteriores_a_la_baja",
            "huecos_internos",
        ]
    ]
