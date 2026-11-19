"""Cofactores: la beta no es un numero, es una funcion del regimen.

El problema que resuelve
------------------------
El Entregable 1 estimo para cada accion una beta contra el mercado::

    rendimiento del activo = alfa + beta * rendimiento del mercado + error

Esa ecuacion impone que la beta sea **la misma siempre**. El mismo numero en
2017, cuando el mercado subia sin sobresaltos, y el 16 de marzo de 2020.

Si la beta en realidad cambia segun las condiciones, ese modelo no esta
simplificando: esta promediando dos regimenes distintos y devolviendo un valor
que no describe ninguno. Y el error va en la direccion peligrosa, porque el
regimen que importa para el riesgo -la crisis- es el menos frecuente, asi que
pesa poco en el promedio.

El modelo con cofactor
----------------------
Un cofactor es una variable que **modifica el efecto de otra** en lugar de
sumar su propio efecto. Se mete como producto::

    r_activo = alfa + beta * r_mercado + gamma * (r_mercado * tension) + error
                            \\________/           \\___________________/
                             efecto base           como lo modifica la tension

Donde ``tension`` vale 1 cuando el mercado esta agitado y 0 cuando esta
tranquilo. Leyendo la ecuacion en cada caso:

* mercado tranquilo (tension = 0):  la beta efectiva es ``beta``
* mercado agitado  (tension = 1):  la beta efectiva es ``beta + gamma``

Entonces ``gamma`` **es** el escalon: cuanto cambia la sensibilidad al pasar de
un regimen a otro. Si gamma sale positiva y significativa, la accion amplifica
mas al mercado en las crisis que en la calma, y el modelo lineal de un solo
coeficiente estaba subestimando su riesgo exactamente cuando mas importaba.

Por que esto produce una relacion no lineal
-------------------------------------------
El termino de interaccion hace que la respuesta al mercado deje de ser una sola
recta y pase a ser dos rectas con pendientes distintas. La relacion resultante
es no lineal en el sentido que importa -- la pendiente depende del estado del
mundo -- pero el modelo sigue siendo lineal en sus parametros, asi que se
estima con minimos cuadrados y cada coeficiente se puede interpretar y someter
a una prueba de significancia. Es la forma mas defendible de medir no
linealidad: nada de caja negra.

De donde sale la definicion de "tension"
----------------------------------------
Partir la historia a mano -por fecha, o con un umbral de rendimiento elegido a
ojo- invitaria a escoger el corte que da el resultado deseado. Para evitarlo, la
tension se define con el GARCH de ``src/volatility.py``: son tensos los dias en
que la **volatilidad condicional del mercado**, estimada con informacion
anterior a ese dia, esta en su cuantil mas alto.

Eso tiene dos ventajas. El corte no lo elige nadie, sale del modelo. Y encadena
los dos modulos del Entregable 3: el GARCH dice *cuando* el mercado esta tenso,
y los cofactores dicen *que le pasa a cada accion* cuando lo esta.

Que parte mira el futuro y que parte no
---------------------------------------
Conviene ser preciso, porque la distincion es facil de pasar por alto. La
volatilidad condicional de cada dia se calcula con la recursion del GARCH, que
usa exclusivamente dias anteriores: esa parte no mira el futuro. El **umbral**
que separa tenso de tranquilo, en cambio, es un cuantil de la serie completa, y
por lo tanto conoce toda la historia.

Para la pregunta que responde este modulo eso es correcto y es lo habitual. Se
trata de una caracterizacion descriptiva -- "en los dias que resultaron tensos,
la beta fue distinta?" -- y no de un pronostico: no se esta prediciendo nada que
luego se evalue, asi que no hay ventaja indebida que contaminar.

Si en cambio se quisiera usar la clasificacion para decidir *hoy* si el mercado
esta tenso, el umbral tendria que calcularse solo con el pasado. Para ese caso
``tension_por_volatilidad`` acepta ``expandiendo=True``, que compara la
volatilidad de cada dia contra el cuantil de los dias previos unicamente.

Errores estandar robustos, y por que no son opcionales
------------------------------------------------------
Los errores estandar de minimos cuadrados de libro suponen que la varianza del
error es constante. Este mismo entregable acaba de demostrar que **no lo es**:
el GARCH encontro persistencia de 0.91 a 0.99 en las cinco acciones probadas.
Usar errores estandar que suponen varianza constante contradiria el hallazgo
anterior del propio proyecto, y en la direccion mala: los subestima, lo que
infla los estadisticos t y hace parecer significativo lo que no lo es.

Por eso se usan los errores estandar robustos a heterocedasticidad de White, en
su variante HC1 con correccion por grados de libertad. El modulo reporta los dos
y la razon entre ellos, para que la diferencia se pueda ver en lugar de
afirmarla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volatility import ajustar_garch, volatilidad_condicional

# Fraccion de dias que se clasifican como tensos. Un quinto deja ~580 dias
# tensos en once anios: suficientes para estimar una pendiente con precision,
# y lo bastante selectivo para que el grupo sean de verdad los dias agitados.
CUANTIL_TENSION = 0.80

# Numero de grupos para la version por escalones. Cinco da ~580 dias por grupo.
N_ESCALONES = 5

# Valor critico de la normal al 5% a dos colas. Con ~2900 observaciones la t de
# Student es indistinguible de la normal.
CRITICO_5PCT = 1.96


# ==========================================================================
# 1. Minimos cuadrados con errores estandar robustos
# ==========================================================================
def _ols_robusto(X: np.ndarray, y: np.ndarray) -> dict:
    """Minimos cuadrados con errores estandar clasicos y robustos (HC1).

    ``X`` debe incluir ya la columna de unos para el intercepto.

    El estimador robusto reemplaza la varianza unica del error por la varianza
    observada dia por dia::

        V = (X'X)^-1  X' diag(e^2) X  (X'X)^-1  *  n / (n - k)

    En lugar de suponer que todos los dias tienen la misma varianza, usa el
    residuo de cada dia como su propia estimacion. El factor ``n / (n - k)`` es
    la correccion por grados de libertad que distingue HC1 de HC0.

    Returns
    -------
    dict con los coeficientes, los dos juegos de errores estandar, los
    estadisticos t robustos y el R cuadrado.
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Hacen falta mas observaciones ({n}) que parametros ({k}).")

    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    residuo = y - X @ coef

    # Clasico: una sola varianza para todos los dias.
    var_residuo = float(residuo @ residuo) / (n - k)
    ee_clasico = np.sqrt(np.maximum(np.diag(XtX_inv) * var_residuo, 0))

    # Robusto HC1: cada dia aporta su propio residuo al cuadrado.
    carne = (X * residuo[:, None] ** 2).T @ X
    V = XtX_inv @ carne @ XtX_inv * (n / (n - k))
    ee_robusto = np.sqrt(np.maximum(np.diag(V), 0))

    centrado = y - y.mean()
    r2 = 1 - float(residuo @ residuo) / float(centrado @ centrado)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_robusto = np.where(ee_robusto > 0, coef / ee_robusto, np.nan)

    return {
        "coef": coef,
        "ee_clasico": ee_clasico,
        "ee_robusto": ee_robusto,
        "t_robusto": t_robusto,
        "r2": r2,
        "n": n,
    }


# ==========================================================================
# 2. Definicion de la tension a partir del GARCH
# ==========================================================================
def mercado(returns: pd.DataFrame) -> pd.Series:
    """Rendimiento del mercado como promedio equiponderado de los activos.

    Es el mismo proxy que usa el Entregable 1: sin capitalizaciones de mercado
    actualizadas dia por dia, el promedio simple es la aproximacion honesta al
    primer componente principal, que es lo que un indice realmente mide.
    """
    return returns.mean(axis=1)


def tension_por_volatilidad(
    r_mercado: pd.Series, cuantil: float = CUANTIL_TENSION,
    parametros: dict | None = None, expandiendo: bool = False,
    minimo: int = 252,
) -> pd.Series:
    """Marca como tensos los dias de mayor volatilidad condicional del mercado.

    La volatilidad de cada dia la estima el GARCH usando solo dias anteriores.
    El umbral, en cambio, depende de ``expandiendo``:

    * ``False`` (por defecto): cuantil de la serie completa. Correcto para la
      pregunta descriptiva de este modulo -- ver la nota del encabezado.
    * ``True``: cuantil calculado con los dias previos unicamente, de modo que
      la clasificacion de cada dia sea reproducible en tiempo real. Los
      primeros ``minimo`` dias quedan como no tensos por falta de historia.
      En esta variante la volatilidad tambien se calcula de forma estrictamente
      causal: se resta una media de cero en lugar de la media muestral, y se
      arranca de la varianza de largo plazo que implican los propios
      parametros, de modo que ningun valor dependa de la longitud de la serie.
      Los parametros del GARCH si se estiman con toda la muestra, como es
      habitual; lo que queda garantizado es que, **dados** los parametros,
      agregar dias al final no reclasifica ningun dia anterior.

    Returns
    -------
    Serie booleana alineada con ``r_mercado``.
    """
    parametros = parametros or ajustar_garch(r_mercado.to_numpy(dtype=np.float64))
    serie = r_mercado.to_numpy(dtype=np.float64)

    if not expandiendo:
        vol = pd.Series(volatilidad_condicional(serie, parametros), index=r_mercado.index)
        return vol >= vol.quantile(cuantil)

    persistencia = parametros["alpha"] + parametros["beta"]
    var_lp = (
        parametros["omega"] / (1 - persistencia) if persistencia < 1 else float(serie.var())
    )
    vol = pd.Series(
        volatilidad_condicional(serie, parametros, media=0.0, var_inicial=var_lp),
        index=r_mercado.index,
    )
    umbral = vol.expanding(min_periods=minimo).quantile(cuantil).shift(1)
    return (vol >= umbral).fillna(False)


def niveles_de_tension(
    r_mercado: pd.Series, n_grupos: int = N_ESCALONES,
    parametros: dict | None = None,
) -> pd.Series:
    """Parte los dias en grupos por volatilidad condicional, de 1 a n_grupos.

    La version de dos regimenes responde "hay escalon?". Esta responde "como es
    la forma del escalon?": si la beta sube monotonamente grupo tras grupo, el
    efecto es gradual y no un salto unico.
    """
    parametros = parametros or ajustar_garch(r_mercado.to_numpy(dtype=np.float64))
    vol = pd.Series(
        volatilidad_condicional(r_mercado.to_numpy(dtype=np.float64), parametros),
        index=r_mercado.index,
    )
    return pd.Series(
        pd.qcut(vol, n_grupos, labels=range(1, n_grupos + 1)).astype(int),
        index=r_mercado.index,
    )


# ==========================================================================
# 3. El modelo con cofactor
# ==========================================================================
def beta_con_interaccion(
    r_activo: pd.Series | np.ndarray,
    r_mercado: pd.Series | np.ndarray,
    tension: pd.Series | np.ndarray,
) -> dict:
    """Estima alfa, beta y gamma del modelo con cofactor.

    Returns
    -------
    dict con los tres coeficientes, la beta efectiva en cada regimen, el
    estadistico t robusto de gamma, si es significativo al 5%, y la razon entre
    el error estandar robusto y el clasico de gamma -- que mide cuanto habria
    exagerado la significancia el calculo de libro.
    """
    y = np.asarray(r_activo, dtype=np.float64)
    m = np.asarray(r_mercado, dtype=np.float64)
    d = np.asarray(tension).astype(np.float64)

    bueno = np.isfinite(y) & np.isfinite(m) & np.isfinite(d)
    y, m, d = y[bueno], m[bueno], d[bueno]
    if len(y) < 100:
        raise ValueError(f"Se necesitan al menos 100 observaciones, hay {len(y)}.")
    if not 0 < d.mean() < 1:
        raise ValueError("La tension debe tener dias de los dos regimenes.")

    # Columnas: intercepto, mercado, mercado*tension.
    # No se incluye la tension sola: su efecto seria un cambio de nivel medio,
    # y lo que se esta midiendo es un cambio de *pendiente*.
    X = np.column_stack([np.ones(len(y)), m, m * d])
    ajuste = _ols_robusto(X, y)
    alfa, beta, gamma = ajuste["coef"]

    razon_ee = (
        ajuste["ee_robusto"][2] / ajuste["ee_clasico"][2]
        if ajuste["ee_clasico"][2] > 0 else np.nan
    )

    return {
        "alfa": float(alfa),
        "beta_calma": float(beta),
        "gamma": float(gamma),
        "beta_tension": float(beta + gamma),
        "cambio_relativo": float(gamma / beta) if beta else np.nan,
        "t_gamma": float(ajuste["t_robusto"][2]),
        "significativo": bool(abs(ajuste["t_robusto"][2]) > CRITICO_5PCT),
        "ee_gamma_robusto": float(ajuste["ee_robusto"][2]),
        "ee_gamma_clasico": float(ajuste["ee_clasico"][2]),
        "razon_ee": float(razon_ee),
        "r2": ajuste["r2"],
        "dias_tensos": int(d.sum()),
        "n": ajuste["n"],
    }


def beta_simple(
    r_activo: pd.Series | np.ndarray, r_mercado: pd.Series | np.ndarray
) -> dict:
    """Beta de un solo coeficiente: el modelo del Entregable 1.

    Se conserva para poder contrastar. La beta unica siempre cae entre las dos
    betas del modelo con cofactor, porque es su promedio ponderado por el numero
    de dias de cada regimen -- y como los dias tensos son minoria, queda mas
    cerca de la beta de calma.
    """
    y = np.asarray(r_activo, dtype=np.float64)
    m = np.asarray(r_mercado, dtype=np.float64)
    bueno = np.isfinite(y) & np.isfinite(m)
    y, m = y[bueno], m[bueno]

    ajuste = _ols_robusto(np.column_stack([np.ones(len(y)), m]), y)
    return {
        "alfa": float(ajuste["coef"][0]),
        "beta": float(ajuste["coef"][1]),
        "t_beta": float(ajuste["t_robusto"][1]),
        "r2": ajuste["r2"],
        "n": ajuste["n"],
    }


def escalones_de_beta(
    r_activo: pd.Series, r_mercado: pd.Series, niveles: pd.Series
) -> pd.DataFrame:
    """Estima la beta por separado dentro de cada nivel de tension.

    Es la version libre del modelo: no impone que el cambio sea un salto unico
    ni que sea lineal en el nivel. Si las betas suben grupo tras grupo, el
    escalon es real y gradual.
    """
    filas = []
    for nivel in sorted(niveles.unique()):
        mascara = niveles == nivel
        y = r_activo[mascara].to_numpy(dtype=np.float64)
        m = r_mercado[mascara].to_numpy(dtype=np.float64)
        bueno = np.isfinite(y) & np.isfinite(m)
        y, m = y[bueno], m[bueno]
        if len(y) < 30:
            continue

        ajuste = _ols_robusto(np.column_stack([np.ones(len(y)), m]), y)
        filas.append({
            "nivel": int(nivel),
            "dias": len(y),
            "beta": float(ajuste["coef"][1]),
            "ee_robusto": float(ajuste["ee_robusto"][1]),
            "r2": ajuste["r2"],
            "vol_mercado": float(m.std() * np.sqrt(252)),
        })
    return pd.DataFrame(filas)


# ==========================================================================
# 4. Recorrido de todo el universo
# ==========================================================================
def tabla_de_cofactores(
    returns: pd.DataFrame,
    r_mercado: pd.Series | None = None,
    tension: pd.Series | None = None,
    sectores: dict | None = None,
) -> pd.DataFrame:
    """Aplica el modelo con cofactor a cada activo del universo.

    Returns
    -------
    DataFrame ordenado por ``gamma`` descendente: arriba quedan las acciones que
    mas amplifican su sensibilidad al mercado cuando el mercado se tensa, que
    son las que el modelo de beta unica describe peor.
    """
    r_mercado = mercado(returns) if r_mercado is None else r_mercado
    tension = tension_por_volatilidad(r_mercado) if tension is None else tension

    filas = []
    for ticker in returns.columns:
        interaccion = beta_con_interaccion(returns[ticker], r_mercado, tension)
        simple = beta_simple(returns[ticker], r_mercado)
        filas.append({
            "ticker": ticker,
            "sector": (sectores or {}).get(ticker, ""),
            "beta_unica": simple["beta"],
            "beta_calma": interaccion["beta_calma"],
            "beta_tension": interaccion["beta_tension"],
            "gamma": interaccion["gamma"],
            "cambio_relativo": interaccion["cambio_relativo"],
            "t_gamma": interaccion["t_gamma"],
            "significativo": interaccion["significativo"],
            "razon_ee": interaccion["razon_ee"],
            "r2_unica": simple["r2"],
            "r2_cofactor": interaccion["r2"],
        })

    return pd.DataFrame(filas).sort_values("gamma", ascending=False).reset_index(drop=True)


def resumen_de_cofactores(tabla: pd.DataFrame) -> pd.DataFrame:
    """Resume la tabla en las cifras que contestan la pregunta del profesor.

    La cifra central es la proporcion de activos con gamma significativa: si es
    alta, el supuesto de beta constante del Entregable 1 falla de forma
    sistematica y no por casos aislados.
    """
    sig = tabla[tabla["significativo"]]
    return pd.DataFrame([
        {"medida": "activos analizados", "valor": float(len(tabla))},
        {"medida": "con gamma significativa al 5%", "valor": float(len(sig))},
        {"medida": "proporcion significativa", "valor": float(tabla["significativo"].mean())},
        {"medida": "de esas, con gamma positiva", "valor": float((sig["gamma"] > 0).mean()) if len(sig) else np.nan},
        {"medida": "beta promedio en calma", "valor": float(tabla["beta_calma"].mean())},
        {"medida": "beta promedio en tension", "valor": float(tabla["beta_tension"].mean())},
        {"medida": "cambio promedio de beta (gamma)", "valor": float(tabla["gamma"].mean())},
        {"medida": "R2 promedio, beta unica", "valor": float(tabla["r2_unica"].mean())},
        {"medida": "R2 promedio, con cofactor", "valor": float(tabla["r2_cofactor"].mean())},
        {"medida": "razon EE robusto / clasico (promedio)", "valor": float(tabla["razon_ee"].mean())},
    ])
