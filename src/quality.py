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


def nulos_por_anio(prices: pd.DataFrame) -> pd.DataFrame:
    """Distribucion de los valores faltantes anio por anio.

    Contar los nulos totales no dice nada por si solo: 54,000 huecos pueden ser
    una fuente rota o pueden ser empresas que todavia no existian. Esta funcion
    responde las dos preguntas que lo distinguen.

    **Primera: como evolucionan?** ``prom_por_dia`` normaliza el conteo por la
    cantidad de dias de mercado de cada anio, que no es constante (2026 esta
    incompleto). Sobre estos datos la serie baja de forma monotona de 41.5
    faltantes por dia en 2015 a 1.3 en 2026. Ese descenso parejo es la evidencia
    de que los huecos son empresas entrando al indice con el tiempo: un fallo de
    la fuente no tendria por que respetar el orden cronologico.

    **Segunda: donde cae el hueco?** Segun su posicion en la historia de cada
    ticker:

    * ``previos_al_listado`` - antes de la primera cotizacion. La accion aun no
      se negociaba; no es un dato perdido.

    * ``posteriores_a_la_baja`` - despues de la ultima cotizacion. La empresa
      salio del indice, fue adquirida o cambio de ticker.

    * ``huecos_internos`` - la empresa ya cotizaba y volvio a cotizar despues,
      pero ese dia no hay precio. **Los unicos preocupantes.** En este panel hay
      exactamente 2 de 54,479, asi que el problema no existe en la practica.

    La suma de las tres columnas es ``n_faltantes``, de modo que el desglose
    siempre cuadra con el total que reporta :func:`structure_summary`.

    Returns
    -------
    DataFrame indexado por anio con: dias_de_mercado, n_faltantes, prom_por_dia,
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
    report["prom_por_dia"] = (
        report["n_faltantes"] / report["dias_de_mercado"]
    ).round(1)
    report["pct_faltantes"] = (
        100 * report["n_faltantes"] / (report["dias_de_mercado"] * prices.shape[1])
    ).round(2)
    report.index.name = "anio"

    return report[
        [
            "dias_de_mercado",
            "n_faltantes",
            "prom_por_dia",
            "pct_faltantes",
            "previos_al_listado",
            "posteriores_a_la_baja",
            "huecos_internos",
        ]
    ]


def calidad_por_sector(
    prices: pd.DataFrame, constituyentes: pd.DataFrame
) -> pd.DataFrame:
    """Cuantos tickers pierde cada sector al exigir historia completa.

    El filtro de ``MIN_OBSERVATIONS`` no cobra su precio de forma uniforme, y eso
    importa porque el analisis sectorial compara sectores entre si. Si uno esta
    medido con menos empresas y con mas huecos que los demas, sus correlaciones
    no son directamente comparables.

    El hallazgo sobre estos datos esta en ``pct_descartados``, no en el conteo
    absoluto. En numeros crudos el sector mas castigado parece Industrials (11
    empresas), pero es tambien el sector mas grande: 11 de 83 es el 13%.
    Communication Services pierde solo 5, pero de 24 - el **21% del sector** - y
    ademas encabeza ``pct_faltante_medio`` con 9.15%. Es decir: el sector con
    menos empresas es justo el que peor medido queda, por las dos vias a la vez.
    Eso se reporta como limitacion del entregable.

    Parameters
    ----------
    prices : panel de precios (fechas x tickers).
    constituyentes : tabla con columnas ``ticker`` y ``sector`` (clasificacion
        GICS), tal como la devuelve :func:`src.data.get_sp500_constituents`.

    Returns
    -------
    DataFrame indexado por sector con: tickers, con_historia_completa,
    descartados, pct_descartados, pct_faltante_medio.
    """
    reporte = per_ticker_report(prices).join(
        constituyentes.set_index("ticker")["sector"]
    )

    # Un ticker sin sector se perderia de forma silenciosa en el groupby, y el
    # total dejaria de cuadrar con el universo. Mejor que sea visible.
    if reporte["sector"].isna().any():
        reporte["sector"] = reporte["sector"].fillna("SIN CLASIFICAR")

    resumen = reporte.groupby("sector").agg(
        tickers=("n_obs", "size"),
        con_historia_completa=("suficiente_historia", "sum"),
        pct_faltante_medio=("pct_missing", "mean"),
    )
    resumen["descartados"] = resumen["tickers"] - resumen["con_historia_completa"]
    resumen["pct_descartados"] = (
        100 * resumen["descartados"] / resumen["tickers"]
    ).round(1)

    return resumen.round(2)[
        [
            "tickers",
            "con_historia_completa",
            "descartados",
            "pct_descartados",
            "pct_faltante_medio",
        ]
    ].sort_values("pct_descartados", ascending=False)


def detectar_precios_estancados(
    prices: pd.DataFrame, dias_minimos: int = 10
) -> pd.DataFrame:
    """Tickers cuyo precio no cambio ni un centavo durante varios dias seguidos.

    Es un tipo de corrupcion distinto al que detecta
    :func:`flag_suspicious_tickers`. Ahi el problema es un precio que se mueve
    demasiado; aqui es uno que no se mueve nada. Una accion del S&P 500 con
    volumen normal cambia de precio todos los dias: que el cierre se repita
    identico durante semanas no es baja volatilidad, es un dato que no se
    actualizo.

    Los tres casos de este panel comparten una explicacion, y no es la obvia:

    * **AMCR** (Amcor), 66 dias congelado en 38.62 dolares entre el 2016-09-06 y
      el 2016-12-08. Amcor plc no cotizaba en NYSE en 2016: empezo el 2019-06-11,
      al completarse la fusion con Bemis. Antes de esa fecha la que cotizaba era
      Amcor Limited, en la bolsa australiana.

    * **SW** (Smurfit Westrock), 34 dias en 16.29 dolares entre el 2016-07-18 y
      el 2016-09-02. La empresa ni existia: nacio de la union de Smurfit Kappa y
      WestRock y debuto en NYSE el 2024-07-08.

    * **FERG** (Ferguson), 19 dias en 49.55 dolares entre el 2015-05-08 y el
      2015-06-05. Su listado en NYSE es del 2021-03-08, y la cotizacion primaria
      se traslado ahi el 2022-05-12; antes operaba en la bolsa de Londres.

    El patron es claro: **en los tres casos el tramo congelado cae anios antes de
    que el ticker existiera en EE.UU.** Yahoo Finance rellena hacia atras la
    historia de estos simbolos con la de la entidad predecesora, y ese relleno
    arrastra tramos donde la cotizacion no se actualizaba - lo esperable en una
    linea poco liquida o en dias en que ese mercado no operaba.

    La consecuencia practica pesa mas que el hallazgo: el precio congelado
    produce rendimientos de exactamente 0.0 durante semanas, y un cero no se
    parece a un dato faltante. Baja la volatilidad estimada del ticker y diluye
    su correlacion con el resto del sector sin que nada lo advierta. El filtro de
    ``MIN_OBSERVATIONS`` no lo atrapa, porque estas celdas no estan vacias.

    Parameters
    ----------
    prices : panel de precios (fechas x tickers).
    dias_minimos : longitud minima de la racha para reportarla. El default de 10
        deja pasar los empates de uno o dos dias, que si pueden ser genuinos.

    Returns
    -------
    DataFrame con: ticker, dias_sin_moverse, fecha_inicio, fecha_fin, precio.
    Vacio (con esas columnas) si no hay ningun caso.
    """
    sin_cambio = (prices.diff() == 0).fillna(False)

    filas = []
    for ticker in prices.columns:
        repetido = sin_cambio[ticker]

        # Agrupar por el cumsum de la negacion asigna un id distinto a cada
        # racha de True consecutivos; el cumsum interno los enumera dentro de
        # la racha, asi que el maximo es la longitud de la racha mas larga.
        rachas = repetido.groupby((~repetido).cumsum()).cumsum()
        largo = int(rachas.max())
        if largo < dias_minimos:
            continue

        fin = rachas.idxmax()
        inicio = prices.index[prices.index.get_loc(fin) - largo]
        filas.append(
            {
                "ticker": ticker,
                "dias_sin_moverse": largo,
                "fecha_inicio": inicio.date(),
                "fecha_fin": fin.date(),
                "precio": round(float(prices[ticker].loc[fin]), 2),
            }
        )

    columnas = ["ticker", "dias_sin_moverse", "fecha_inicio", "fecha_fin", "precio"]
    if not filas:
        # Sin esto, un dias_minimos alto haria fallar el sort_values con
        # KeyError sobre un DataFrame sin columnas.
        return pd.DataFrame(columns=columnas)

    return pd.DataFrame(filas).sort_values("dias_sin_moverse", ascending=False)
