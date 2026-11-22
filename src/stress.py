"""Pruebas de estres sobre una cartera.

El problema que resuelve
------------------------
Los modulos anteriores describen el riesgo **en promedio**: una volatilidad
anualizada, una matriz de covarianza, una beta. Son la respuesta correcta a la
pregunta "cuanto se mueve esto normalmente", y la pregunta que de verdad
importa es otra:

    Que le pasa a esta cartera si se repite marzo de 2020?

Un promedio de once anios no contesta eso, y el Entregable 2 midio por cuanto
se equivoca: el motor calibrado con toda la historia subestima el riesgo un
29% durante las crisis, justo cuando alguien lo consultaria.

Este modulo cierra ese hueco con tres enfoques que se complementan.

1. Reproduccion de episodios reales
-----------------------------------
El mas simple y el mas dificil de discutir: se toma la cartera actual, se le
aplican los rendimientos que **de verdad ocurrieron** en una ventana historica,
y se reporta lo que habria pasado. No hay supuestos distribucionales ni
parametros que calibrar; solo aritmetica sobre datos observados.

Su limitacion es evidente y conviene decirla: el pasado no agota el espacio de
lo posible, y la peor caida de la muestra no es la peor caida concebible.

2. Riesgo condicional al regimen
--------------------------------
En lugar de una sola matriz de covarianza para toda la historia, se calcula una
con los dias de calma y otra con los dias de tension -clasificados por el GARCH
de ``volatility.py``- y se compara el riesgo de la misma cartera bajo cada una.
La diferencia mide cuanta proteccion desaparece cuando hace falta.

3. Proyeccion ante un choque de mercado
---------------------------------------
Dado un movimiento del mercado, se proyecta el efecto sobre la cartera de dos
maneras: con la beta unica del Entregable 1 y con la beta condicional al
regimen de ``cofactors.py``. La diferencia entre ambas **no es un detalle
tecnico**: es, en puntos de rendimiento, el error de suponer que la beta no
cambia. Es la aplicacion directa del hallazgo central del Entregable 3.

Medidas de perdida
------------------
Se reportan VaR y Expected Shortfall, ambos **historicos**: se leen de la
distribucion observada en lugar de suponer normalidad. Es deliberado. Este
mismo proyecto documento que los rendimientos tienen colas mas pesadas que la
normal -es la razon por la que el GARCH subestima los picos de crisis- asi que
un VaR parametrico gaussiano heredaria ese error justo en la cola que pretende
medir.

El Expected Shortfall acompania al VaR y no al reves. El VaR dice cuanto se
pierde en el peor 5% de los dias **en el mejor de esos casos**; no dice nada
sobre que tan grave puede ser el resto. El Expected Shortfall promedia toda la
cola, y por eso es la medida que el marco de Basilea adopto en sustitucion del
VaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk import DIAS_HABILES, risk_decomposition, sample_covariance

# Episodios de tension verificados sobre el panel 2015-2026. Las fechas son
# las de inicio y fin del tramo de caida, no del evento en sentido amplio.
ESCENARIOS: dict[str, tuple[str, str]] = {
    "COVID-19": ("2020-02-19", "2020-03-23"),
    "Mercado bajista 2022": ("2022-01-03", "2022-10-12"),
    "Cuarto trimestre 2018": ("2018-10-01", "2018-12-24"),
    "Volmageddon 2018": ("2018-01-26", "2018-02-08"),
    "Desarme del carry 2024": ("2024-07-31", "2024-08-05"),
}

NIVEL_VAR = 0.95


# ==========================================================================
# 1. Serie de la cartera
# ==========================================================================
def serie_portafolio(
    returns: pd.DataFrame, pesos: np.ndarray, tickers: list[str]
) -> pd.Series:
    """Rendimiento diario de una cartera de pesos fijos.

    La suma ponderada se hace sobre rendimientos **simples**, no logaritmicos.
    El detalle importa: el logaritmo no es lineal, asi que el rendimiento
    logaritmico de una cartera no es el promedio ponderado de los logaritmicos
    de sus posiciones. A un dia la diferencia es de segundo orden, pero al
    acumular una ventana de crisis se vuelve visible, y es justamente la
    ventana que este modulo mide.

    Returns
    -------
    Serie de rendimientos logaritmicos de la cartera, para que sumen al
    acumular.
    """
    w = np.asarray(pesos, dtype=np.float64)
    if len(w) != len(tickers):
        raise ValueError(f"{len(w)} pesos para {len(tickers)} activos.")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1, suman {w.sum():.6f}.")

    simples = np.expm1(returns[tickers].to_numpy(dtype=np.float64))
    return pd.Series(np.log1p(simples @ w), index=returns.index, name="portafolio")


# ==========================================================================
# 2. Medidas de perdida
# ==========================================================================
def var_historico(serie: pd.Series, nivel: float = NIVEL_VAR) -> dict:
    """VaR y Expected Shortfall historicos a un dia.

    Ambos se leen directamente de la distribucion observada, sin suponer
    normalidad. Se devuelven como **perdidas positivas**: un VaR de 2.1%
    significa que se espera perder 2.1% o mas en el 5% de los dias.

    Returns
    -------
    dict con el VaR, el Expected Shortfall, la razon entre ambos y el numero
    de dias que componen la cola. La razon ES/VaR mide cuanto mas grave es la
    cola de lo que el VaR por si solo sugiere.
    """
    if not 0.5 < nivel < 1:
        raise ValueError(f"El nivel debe estar entre 0.5 y 1, es {nivel}.")

    simples = np.expm1(serie.dropna().to_numpy(dtype=np.float64))
    if len(simples) < 100:
        raise ValueError(f"Se necesitan al menos 100 observaciones, hay {len(simples)}.")

    corte = float(np.quantile(simples, 1 - nivel))
    cola = simples[simples <= corte]

    var = -corte
    es = -float(cola.mean())
    return {
        "nivel": nivel,
        "var": var,
        "expected_shortfall": es,
        "razon_es_var": es / var if var > 0 else np.nan,
        "dias_en_la_cola": int(len(cola)),
        "n": len(simples),
    }


def maxima_caida(serie: pd.Series) -> dict:
    """Maxima caida acumulada desde un maximo previo (drawdown).

    Es la medida que un inversionista reconoce de inmediato, porque es la
    perdida que habria visto en su estado de cuenta: no la de un dia suelto,
    sino la acumulada desde el punto mas alto hasta el mas bajo posterior.
    """
    s = serie.dropna()
    if s.empty:
        raise ValueError("La serie esta vacia.")

    acumulado = s.cumsum()
    maximo_previo = acumulado.cummax()
    caida = np.expm1(acumulado - maximo_previo)

    fondo = caida.idxmin()
    pico = acumulado.loc[:fondo].idxmax()
    posterior = caida.loc[fondo:]
    recuperada = posterior[posterior >= -1e-9]

    return {
        "caida": float(caida.min()),
        "pico": pico,
        "fondo": fondo,
        "dias_de_caida": int(len(acumulado.loc[pico:fondo]) - 1),
        "recuperacion": recuperada.index[0] if len(recuperada) else None,
        "dias_de_recuperacion": (
            int(len(posterior.loc[:recuperada.index[0]]) - 1) if len(recuperada) else None
        ),
    }


# ==========================================================================
# 3. Reproduccion de episodios reales
# ==========================================================================
def escenario_historico(
    serie: pd.Series, inicio: str, fin: str, etiqueta: str = ""
) -> dict:
    """Aplica a la cartera los rendimientos realmente ocurridos en una ventana.

    No hay modelo de por medio: es la cartera de hoy sometida a los movimientos
    de un periodo que ya paso.
    """
    tramo = serie.loc[inicio:fin].dropna()
    if tramo.empty:
        raise ValueError(f"No hay datos entre {inicio} y {fin}.")

    return {
        "escenario": etiqueta or f"{inicio} a {fin}",
        "inicio": tramo.index[0],
        "fin": tramo.index[-1],
        "sesiones": len(tramo),
        "rendimiento": float(np.expm1(tramo.sum())),
        "peor_dia": float(np.expm1(tramo.min())),
        "fecha_peor_dia": tramo.idxmin(),
        "volatilidad": float(tramo.std() * np.sqrt(DIAS_HABILES)),
        "caida_maxima": maxima_caida(tramo)["caida"],
    }


def tabla_de_escenarios(
    serie: pd.Series, escenarios: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Aplica todos los escenarios y los ordena del mas severo al menos."""
    filas = []
    for etiqueta, (a, b) in (escenarios or ESCENARIOS).items():
        try:
            filas.append(escenario_historico(serie, a, b, etiqueta))
        except ValueError:
            continue  # el escenario cae fuera del periodo disponible
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values("rendimiento").reset_index(drop=True)


def peores_ventanas(
    serie: pd.Series, dias: int = 21, n: int = 5, separacion: int = 90
) -> pd.DataFrame:
    """Encuentra las peores ventanas moviles sin que se traslapen.

    Complementa a los escenarios con nombre, que fueron elegidos por una
    persona. Estas salen de los datos: si el peor episodio del periodo no
    coincide con ninguno de los nombrados, aparece aqui de todos modos.

    ``separacion`` evita que las cinco respuestas sean cinco recortes del mismo
    desplome.
    """
    acumulado = serie.rolling(dias).sum().dropna()
    filas, usadas = [], []

    for fecha, valor in acumulado.sort_values().items():
        if any(abs((fecha - u).days) < separacion for u in usadas):
            continue
        pos = serie.index.get_loc(fecha)
        filas.append({
            "inicio": serie.index[max(0, pos - dias + 1)],
            "fin": fecha,
            "rendimiento": float(np.expm1(valor)),
        })
        usadas.append(fecha)
        if len(filas) == n:
            break

    return pd.DataFrame(filas)


# ==========================================================================
# 4. Riesgo condicional al regimen
# ==========================================================================
def riesgo_por_regimen(
    returns: pd.DataFrame,
    pesos: np.ndarray,
    tickers: list[str],
    tension: pd.Series,
) -> pd.DataFrame:
    """Compara la misma cartera bajo las covarianzas de calma y de tension.

    Es la prueba que desmonta el supuesto del motor de riesgo. Si el ratio de
    diversificacion baja en los dias de tension, las correlaciones subieron, y
    las mismas posiciones protegen menos exactamente cuando se necesita que
    protejan.

    Parameters
    ----------
    tension
        Serie booleana alineada con ``returns``, normalmente la que produce
        ``cofactors.tension_por_volatilidad``.
    """
    w = np.asarray(pesos, dtype=np.float64)
    tension = tension.reindex(returns.index).fillna(False).to_numpy(dtype=bool)

    filas = []
    for nombre, mascara in [
        ("Periodo completo", np.ones(len(returns), dtype=bool)),
        ("Calma", ~tension),
        ("Tensión", tension),
    ]:
        sub = returns.loc[mascara, tickers].to_numpy(dtype=np.float64)
        if len(sub) < len(tickers) + 2:
            continue
        r = risk_decomposition(sample_covariance(sub), w, tickers)
        filas.append({
            "regimen": nombre,
            "sesiones": len(sub),
            "vol_portafolio": r["portfolio_vol"],
            "vol_ponderada": r["weighted_avg_vol"],
            "ratio_diversificacion": r["diversification_ratio"],
        })

    tabla = pd.DataFrame(filas)
    base = tabla.loc[tabla["regimen"] == "Periodo completo", "vol_portafolio"]
    if len(base):
        tabla["vs_completo"] = tabla["vol_portafolio"] / float(base.iloc[0]) - 1
    return tabla


# ==========================================================================
# 5. Proyeccion ante un choque de mercado
# ==========================================================================
def choque_de_mercado(
    pesos: np.ndarray,
    tickers: list[str],
    cofactores: pd.DataFrame,
    choque: float = -0.10,
) -> dict:
    """Proyecta el efecto de un movimiento del mercado sobre la cartera.

    Lo hace dos veces, con los dos modelos que el proyecto estimo:

    * **Beta unica**, la del Entregable 1: un solo coeficiente por activo para
      toda la historia.
    * **Beta condicional**, la del Entregable 3: ``beta + gamma``, la que
      aplica cuando el mercado esta en tension.

    La diferencia entre ambas proyecciones es el error, medido en puntos de
    rendimiento, de suponer que la beta no cambia con el regimen. Como un
    choque de mercado es precisamente un episodio de tension, la segunda es la
    que corresponde usar, y la primera es la que un modelo estandar usaria.

    Dos cantidades que conviene no confundir
    ----------------------------------------
    ``gamma`` es el salto de la beta de **calma** a la de tension, por
    definicion del modelo con cofactor: ``beta_tension = beta_calma + gamma``.

    El error del modelo estandar es otra cosa y es **menor**, porque la beta
    unica no es la de calma: se estima sobre todos los dias, de los cuales una
    quinta parte son de tension, asi que ya absorbe una parte del salto. El
    error que de verdad se comete al usarla es::

        beta_tension - beta_unica   <   gamma

    El modulo devuelve las dos cifras por separado. Reportar ``gamma`` como si
    fuera el error exagerararia el problema, y este proyecto no necesita
    exagerarlo.

    Parameters
    ----------
    cofactores
        Tabla de ``cofactors.tabla_de_cofactores``, indexada por ticker en la
        columna ``ticker``.
    choque
        Movimiento del mercado como rendimiento simple. Negativo para caidas.
    """
    w = np.asarray(pesos, dtype=np.float64)
    tabla = cofactores.set_index("ticker")

    faltan = [tk for tk in tickers if tk not in tabla.index]
    if faltan:
        raise ValueError(f"Sin beta condicional estimada para: {', '.join(faltan)}.")

    sub = tabla.loc[tickers]
    beta_unica = float(w @ sub["beta_unica"].to_numpy())
    beta_calma = float(w @ sub["beta_calma"].to_numpy())
    beta_tension = float(w @ sub["beta_tension"].to_numpy())
    gamma = float(w @ sub["gamma"].to_numpy())

    perdida_estandar = beta_unica * choque
    perdida_condicional = beta_tension * choque

    return {
        "choque": choque,
        "beta_unica": beta_unica,
        "beta_calma": beta_calma,
        "beta_tension": beta_tension,
        # El salto completo del modelo con cofactor, de calma a tension.
        "gamma_cartera": gamma,
        # Lo que de verdad se le escapa a la beta unica, que es menor que
        # gamma porque esa beta ya incorpora los dias de tension de la muestra.
        "salto_vs_beta_unica": beta_tension - beta_unica,
        "perdida_modelo_estandar": perdida_estandar,
        "perdida_modelo_condicional": perdida_condicional,
        "error_del_modelo_estandar": perdida_condicional - perdida_estandar,
        "subestimacion_relativa": (
            perdida_condicional / perdida_estandar - 1 if perdida_estandar else np.nan
        ),
        "posiciones": len(tickers),
    }


def detalle_del_choque(
    pesos: np.ndarray,
    tickers: list[str],
    cofactores: pd.DataFrame,
    choque: float = -0.10,
) -> pd.DataFrame:
    """Desglosa la proyeccion del choque posicion por posicion.

    Permite ver que posiciones concentran el error: son las que combinan peso
    alto con un salto grande de beta, no necesariamente las mas volatiles.

    La columna ``error`` usa ``beta_tension - beta_unica`` y no ``gamma``, por
    la razon que explica ``choque_de_mercado``: son cantidades distintas, y
    esta es la que suma al error agregado.
    """
    w = np.asarray(pesos, dtype=np.float64)
    sub = cofactores.set_index("ticker").loc[tickers]
    unica = sub["beta_unica"].to_numpy()
    tension = sub["beta_tension"].to_numpy()

    return pd.DataFrame({
        "ticker": tickers,
        "sector": sub["sector"].to_numpy(),
        "peso": w,
        "beta_unica": unica,
        "beta_tension": tension,
        "gamma": sub["gamma"].to_numpy(),
        "aporte_estandar": w * unica * choque,
        "aporte_condicional": w * tension * choque,
        "error": w * (tension - unica) * choque,
    }).sort_values("error").reset_index(drop=True)
