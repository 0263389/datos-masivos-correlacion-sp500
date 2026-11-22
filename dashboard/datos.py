"""Carga y calculos compartidos del tablero.

Separar esta capa de la interfaz tiene un proposito concreto: todo lo caro se
calcula aqui una sola vez y queda en memoria. Sin eso, cada clic en el menu
volveria a ajustar el GARCH del mercado y a recorrer medio millon de pares.

Ninguna de estas funciones contiene logica de analisis. Todas delegan en los
modulos de ``src/``, que son los que se prueban y los que se defienden; aqui
solo se orquesta y se memoriza.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import cofactors as cf  # noqa: E402
from src import nonlinear as nl  # noqa: E402
from src.analysis import (  # noqa: E402
    correlation_matrix,
    distance_matrix,
    pair_table,
    sector_comparison,
)
from src.data import get_prices, get_sp500_constituents  # noqa: E402
from src.returns import to_log_returns  # noqa: E402
from src.risk import (  # noqa: E402
    equal_weights,
    ledoit_wolf_shrinkage,
    min_variance_weights_long_only,
    risk_decomposition,
    sample_covariance,
)
from src.network import (  # noqa: E402
    arbol_de_expansion_minima,
    aristas_por_sector,
    disposicion,
    grados,
    pureza_sectorial,
)
from src.stress import (  # noqa: E402
    choque_de_mercado,
    detalle_del_choque,
    maxima_caida,
    peores_ventanas,
    riesgo_por_regimen,
    serie_portafolio,
    tabla_de_escenarios,
    var_historico,
)
from src.validation import barrido_de_ventanas  # noqa: E402
from src.volatility import (  # noqa: E402
    ajustar_garch,
    predecir_volatilidad,
    volatilidad_condicional,
)

# Una hora de vida para los datos: son precios de cierre, no cambian intradia.
TTL = 3600

# Asignaciones disponibles. La comparacion se hace contra estas constantes y
# nunca contra una cadena escrita a mano: una diferencia de mayusculas entre la
# interfaz y el calculo haria que una opcion cayera silenciosamente en la otra.
MINIMA_VARIANZA = "Mínima varianza"
EQUIPONDERADO = "Equiponderado"


def _equipondera(metodo: str) -> bool:
    """Normaliza el nombre de la asignacion recibido desde la interfaz."""
    return metodo.strip().lower().startswith("equi")


# ==========================================================================
# 1. Datos base
# ==========================================================================
@st.cache_data(ttl=TTL, show_spinner="Cargando precios...")
def cargar() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Panel de rendimientos balanceado y el mapa de sectores.

    Se descartan las columnas con cualquier hueco en lugar de rellenarlas. Es
    la misma decision del Entregable 1: rellenar un precio faltante inventa un
    dia de rendimiento cero, que es informacion falsa y contamina la matriz de
    correlacion.
    """
    cons = get_sp500_constituents()
    sectores = dict(zip(cons["ticker"], cons["sector"]))
    rets = to_log_returns(get_prices([])).dropna(axis=1, how="any").dropna()
    return rets, cons, sectores


@st.cache_data(ttl=TTL)
def mercado_y_garch(_rets_id: str) -> tuple[pd.Series, dict, pd.Series, pd.Series]:
    """Mercado equiponderado, su GARCH, la volatilidad condicional y la tension.

    Es el calculo mas caro del tablero y el que mas se reutiliza: la volatilidad
    condicional del mercado alimenta la pagina de volatilidad y define el
    regimen que usan los cofactores.
    """
    rets, _, _ = cargar()
    m = cf.mercado(rets)
    params = ajustar_garch(m.to_numpy(dtype=np.float64))
    vol = pd.Series(volatilidad_condicional(m.to_numpy(dtype=np.float64), params),
                    index=m.index)
    tension = vol >= vol.quantile(cf.CUANTIL_TENSION)
    return m, params, vol, tension


@st.cache_data(ttl=TTL, show_spinner="Ajustando GARCH...")
def garch_de(ticker: str) -> tuple[dict, pd.Series, np.ndarray]:
    """Ajuste, volatilidad condicional historica y pronostico a un anio."""
    rets, _, _ = cargar()
    serie = rets[ticker].to_numpy(dtype=np.float64)
    params = ajustar_garch(serie)
    vol = pd.Series(volatilidad_condicional(serie, params), index=rets.index)
    return params, vol, predecir_volatilidad(serie, params, horizonte=252)


# ==========================================================================
# 2. Correlaciones (Entregable 1)
# ==========================================================================
@st.cache_data(ttl=TTL, show_spinner="Midiendo pares...")
def pares_y_sectores(n_activos: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rets, cons, sectores = cargar()
    uni = _muestra(rets.columns, n_activos)
    pares = pair_table(correlation_matrix(rets[uni]), cons)
    return pares, sector_comparison(pares)


@st.cache_data(ttl=TTL)
def matriz_por_sector(n_activos: int) -> pd.DataFrame:
    rets, cons, sectores = cargar()
    uni = _muestra(rets.columns, n_activos)
    corr = correlation_matrix(rets[uni])
    orden = sorted(uni, key=lambda t: (sectores.get(t, ""), t))
    return corr.loc[orden, orden]


# ==========================================================================
# 3. Riesgo (Entregable 2)
# ==========================================================================
@st.cache_data(ttl=TTL)
def riesgo(tickers: tuple[str, ...], metodo: str, encoger: bool) -> tuple[dict, float]:
    """Descomposicion de riesgo del portafolio que arme el usuario.

    Devuelve tambien ``delta``, la intensidad de encogimiento que eligio
    Ledoit-Wolf: es informativa en si misma, porque crece cuando hay pocos
    datos por activo y el estimador muestral es poco confiable.
    """
    rets, _, _ = cargar()
    X = rets[list(tickers)].to_numpy(dtype=np.float64)
    if encoger:
        cov, delta, _ = ledoit_wolf_shrinkage(X)
    else:
        cov, delta = sample_covariance(X), 0.0
    pesos = (
        equal_weights(len(tickers)) if _equipondera(metodo)
        else min_variance_weights_long_only(cov)
    )
    return risk_decomposition(cov, pesos, list(tickers)), float(delta)


@st.cache_data(ttl=TTL)
def _pesos_de(tickers: tuple[str, ...], metodo: str, encoger: bool) -> np.ndarray:
    """Los pesos de la cartera, sin la descomposicion de riesgo completa."""
    rets, _, _ = cargar()
    X = rets[list(tickers)].to_numpy(dtype=np.float64)
    cov = ledoit_wolf_shrinkage(X)[0] if encoger else sample_covariance(X)
    return (
        equal_weights(len(tickers)) if _equipondera(metodo)
        else min_variance_weights_long_only(cov)
    )


@st.cache_data(ttl=TTL, show_spinner="Corriendo validacion hacia adelante...")
def barrido(n_activos: int) -> pd.DataFrame:
    rets, _, _ = cargar()
    uni = _muestra(rets.columns, n_activos)
    return barrido_de_ventanas(
        rets[uni], ventanas=[125, 250, 500, 1000, 1500, 2000],
        metodo="min_var_largo", encoger=True,
    )


# ==========================================================================
# 4. No linealidad y cofactores (Entregable 3)
# ==========================================================================
@st.cache_data(ttl=TTL, show_spinner="Construyendo la red del mercado...")
def red(n_activos: int) -> dict:
    """Esqueleto del mercado: arbol minimo, concentradores y pureza sectorial.

    Se agrupa en un solo calculo porque todo parte del mismo arbol, y la
    disposicion de los nodos es lo unico que cuesta -- un segundo y medio para
    458 activos.
    """
    rets, _, sectores = cargar()
    uni = _muestra(rets.columns, n_activos)
    distancias = distance_matrix(correlation_matrix(rets[uni]))
    arbol = arbol_de_expansion_minima(distancias)
    return {
        "arbol": arbol,
        "grados": grados(arbol, sectores),
        "pureza": pureza_sectorial(arbol, sectores),
        "por_sector": aristas_por_sector(arbol, sectores),
        "posiciones": disposicion(arbol),
        "sectores": sectores,
        "n_posibles": len(uni) * (len(uni) - 1) // 2,
    }


@st.cache_data(ttl=TTL, show_spinner="Midiendo dependencia no lineal...")
def no_lineal(n_activos: int) -> tuple[pd.DataFrame, float]:
    rets, cons, sectores = cargar()
    uni = _muestra(rets.columns, n_activos)
    sub = rets[uni]
    return nl.tabla_no_lineal(sub, sectores=sectores), nl.piso_de_brecha(sub)["percentil_95"]


@st.cache_data(ttl=TTL, show_spinner="Estimando betas por regimen...")
def cofactores() -> pd.DataFrame:
    rets, cons, sectores = cargar()
    m, _, _, tension = mercado_y_garch("v1")
    return cf.tabla_de_cofactores(rets, m, tension, sectores)


@st.cache_data(ttl=TTL, show_spinner="Aplicando escenarios de estres...")
def estres(tickers: tuple[str, ...], metodo: str, encoger: bool,
           choque: float) -> dict:
    """Todas las medidas de estres de una cartera, en un solo calculo.

    Se agrupan aqui porque comparten los mismos pesos y la misma serie: si
    cada una los recalculara por separado, cambiar el choque en la interfaz
    volveria a ajustar el GARCH del mercado.
    """
    rets, _, _ = cargar()
    lista = list(tickers)
    _, delta = riesgo(tickers, metodo, encoger)
    pesos = _pesos_de(tickers, metodo, encoger)

    serie = serie_portafolio(rets, pesos, lista)
    _, _, _, tension = mercado_y_garch("v1")
    cof = cofactores()

    return {
        "pesos": pesos,
        "delta": delta,
        "serie": serie,
        "var95": var_historico(serie, 0.95),
        "var99": var_historico(serie, 0.99),
        "caida": maxima_caida(serie),
        "escenarios": tabla_de_escenarios(serie),
        "ventanas": peores_ventanas(serie),
        "regimenes": riesgo_por_regimen(rets, pesos, lista, tension),
        "choque": choque_de_mercado(pesos, lista, cof, choque),
        "detalle": detalle_del_choque(pesos, lista, cof, choque),
    }


@st.cache_data(ttl=TTL)
def escalones(ticker: str, n_grupos: int) -> pd.DataFrame:
    rets, _, _ = cargar()
    m, params, _, _ = mercado_y_garch("v1")
    niveles = cf.niveles_de_tension(m, n_grupos, parametros=params)
    return cf.escalones_de_beta(rets[ticker], m, niveles)


# ==========================================================================
# 5. Utilidades
# ==========================================================================
def _muestra(columnas, n: int) -> list[str]:
    """Submuestra reproducible del universo.

    La semilla es fija a proposito: si cambiara entre clics, los numeros del
    tablero cambiarian sin que el usuario haya tocado nada.
    """
    columnas = list(columnas)
    if n >= len(columnas):
        return sorted(columnas)
    return sorted(np.random.default_rng(0).choice(columnas, n, replace=False).tolist())
