"""Relaciones no lineales entre activos.

El problema que resuelve
------------------------
Los Entregables 1 y 2 midieron todo con correlacion. La correlacion de Pearson
es, por construccion, la pendiente de la mejor **recta** que pasa por la nube de
puntos. Si la relacion entre dos activos no es una recta, la correlacion la
reporta mal o no la reporta.

El caso que importa en finanzas no es exotico. Dos acciones pueden moverse de
forma independiente durante anios y desplomarse juntas en una crisis. La
correlacion promedia los dos regimenes y devuelve un numero intermedio que no
describe ninguno de los dos: subestima el riesgo cuando mas importa y lo
sobrestima el resto del tiempo.

Este modulo mide dependencia **sin suponer que es lineal**, con dos
herramientas que atacan el problema desde angulos distintos.

1. Informacion mutua: cuanta dependencia hay, de cualquier forma
----------------------------------------------------------------
La informacion mutua mide cuanto se reduce la incertidumbre sobre un activo al
conocer el otro. Vale cero si y solo si son independientes, y no supone ninguna
forma funcional: detecta relaciones en U, en escalon, o cualquier otra.

Se estima discretizando las dos series y comparando la tabla conjunta contra el
producto de las marginales::

    MI = suma  p(i,j) * log( p(i,j) / (p(i) * p(j)) )

Los cortes se hacen por **cuantiles** y no por intervalos de igual ancho. Con
intervalos de igual ancho, los rendimientos financieros -que tienen colas
gordas- dejarian casi todas las observaciones en la casilla central y las colas
vacias, que es justamente donde esta la informacion. Por cuantiles cada casilla
recibe el mismo numero de observaciones.

El sesgo que hay que restar
---------------------------
La informacion mutua estimada de una muestra finita **nunca sale cero**, ni
siquiera con datos independientes: cada casilla se desvia un poco de su valor
esperado por azar, y esas desviaciones se suman siempre en la misma direccion.
El valor esperado de ese sesgo bajo independencia es conocido::

    sesgo = (bins - 1)^2 / (2 * n)

Reportar la informacion mutua cruda sin restar esto haria "encontrar"
dependencia entre cualquier par de series. El modulo la resta por defecto, y
``sesgo_por_permutacion`` permite verificar empiricamente que la correccion
analitica es del tamanio correcto -- revolviendo una de las dos series se
destruye cualquier relacion real y lo que queda es puro sesgo.

De informacion mutua a una escala interpretable
-----------------------------------------------
La informacion mutua se mide en nats y no tiene tope, asi que un valor de 0.08
no dice nada por si solo. Para hacerla comparable con la correlacion se usa la
identidad que vale para una normal bivariada::

    MI = -0.5 * log(1 - rho^2)      =>      rho = sqrt(1 - exp(-2 * MI))

El resultado se lee como "la correlacion que tendria que haber para producir
esta cantidad de dependencia". Y de ahi sale la medida central del modulo::

    brecha no lineal = rho equivalente - |rho de Pearson|

Si la brecha es cerca de cero, la correlacion ya capturaba toda la dependencia y
el analisis lineal de los entregables anteriores era suficiente para ese par. Si
la brecha es grande, hay estructura que la correlacion no ve.

2. Dependencia de cola: se caen juntas?
---------------------------------------
La informacion mutua dice *cuanta* dependencia hay, pero no *donde* esta. Para
el riesgo de un portafolio la pregunta precisa es otra: si este activo tiene uno
de sus peores dias, cual es la probabilidad de que el otro tambien?

Eso se mide contando directamente::

    P(Y en su peor 10% | X en su peor 10%)

Bajo independencia esa probabilidad seria 10%. Si sale 35%, las caidas estan
tres veces y media mas sincronizadas de lo que serian por azar.

La parte que la correlacion no puede expresar es la **asimetria**. La
correlacion es un solo numero para toda la nube de puntos: no puede decir que
las caidas van mas juntas que las subidas. Comparar la cola inferior contra la
superior si lo dice, y es un hecho documentado de los mercados -- la
diversificacion se desvanece justo en las crisis, que es cuando se necesita.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numero de casillas por variable. Con 8 casillas y ~2900 observaciones quedan
# unas 45 observaciones por celda de la tabla conjunta de 8x8, suficiente para
# que las frecuencias sean estables. Mas casillas detectan estructura mas fina
# pero el sesgo crece con el cuadrado de este numero.
BINS = 8

# Fraccion que define "evento extremo" en la dependencia de cola. Con 10% y
# ~2900 dias quedan ~290 dias extremos, bastantes para estimar una probabilidad
# condicional con precision razonable.
CUANTIL_COLA = 0.10


# ==========================================================================
# 1. Informacion mutua
# ==========================================================================
def _indices_por_cuantiles(x: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Asigna a cada observacion su casilla, usando cortes por cuantiles.

    Devuelve enteros de 0 a ``bins - 1``. Si la serie tiene valores repetidos
    -por ejemplo muchos ceros- algunos cortes coinciden y las casillas
    resultantes quedan desiguales o vacias. Eso no rompe el calculo: una casilla
    vacia aporta cero a la informacion mutua.
    """
    bordes = np.quantile(x, np.linspace(0, 1, bins + 1))[1:-1]
    return np.digitize(x, bordes)


def informacion_mutua(
    x: np.ndarray, y: np.ndarray, bins: int = BINS, corregir_sesgo: bool = True
) -> float:
    """Informacion mutua entre dos series, en nats.

    Vale cero si las series son independientes y crece con la dependencia, sea
    lineal o no.

    Parameters
    ----------
    corregir_sesgo
        Resta el valor esperado bajo independencia. Dejarlo en ``False`` sirve
        para comparar contra ``sesgo_por_permutacion``, no para reportar.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bueno = np.isfinite(x) & np.isfinite(y)
    x, y = x[bueno], y[bueno]
    n = len(x)
    if n < 100:
        raise ValueError(f"Se necesitan al menos 100 observaciones, hay {n}.")

    return _mi_desde_indices(
        _indices_por_cuantiles(x, bins), _indices_por_cuantiles(y, bins),
        bins, n, corregir_sesgo,
    )


def _mi_desde_indices(
    ix: np.ndarray, iy: np.ndarray, bins: int, n: int, corregir_sesgo: bool
) -> float:
    """Nucleo del calculo, ya con las series discretizadas.

    Esta separacion existe por eficiencia: al recorrer muchos pares, cada serie
    se discretiza **una sola vez** y aqui solo se arma la tabla conjunta.
    """
    conjunta = np.bincount(ix * bins + iy, minlength=bins * bins).reshape(bins, bins)
    p = conjunta / n
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)

    hay = p > 0
    mi = float(np.sum(p[hay] * np.log(p[hay] / (px * py)[hay])))

    if corregir_sesgo:
        mi -= (bins - 1) ** 2 / (2 * n)
    return max(mi, 0.0)


def sesgo_por_permutacion(
    x: np.ndarray, y: np.ndarray, bins: int = BINS, repeticiones: int = 200,
    semilla: int = 0,
) -> dict:
    """Mide empiricamente el sesgo revolviendo una de las dos series.

    Revolver destruye cualquier relacion real pero conserva las distribuciones
    marginales, asi que la informacion mutua que queda es puro sesgo de muestra
    finita. Sirve para comprobar que la correccion analitica
    ``(bins - 1)^2 / (2n)`` es del tamanio correcto, y para poner la
    informacion mutua observada en contexto.

    Returns
    -------
    dict con la MI observada sin corregir, la media y desviacion del sesgo
    empirico, el sesgo analitico, y el puntaje z de la observacion contra la
    distribucion de permutaciones.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bueno = np.isfinite(x) & np.isfinite(y)
    x, y = x[bueno], y[bueno]
    n = len(x)

    ix = _indices_por_cuantiles(x, bins)
    iy = _indices_por_cuantiles(y, bins)
    observada = _mi_desde_indices(ix, iy, bins, n, corregir_sesgo=False)

    rng = np.random.default_rng(semilla)
    nulas = np.array([
        _mi_desde_indices(ix, rng.permutation(iy), bins, n, corregir_sesgo=False)
        for _ in range(repeticiones)
    ])

    return {
        "mi_observada": observada,
        "sesgo_empirico": float(nulas.mean()),
        "desviacion_empirica": float(nulas.std()),
        "sesgo_analitico": (bins - 1) ** 2 / (2 * n),
        "z": float((observada - nulas.mean()) / nulas.std()) if nulas.std() > 0 else np.inf,
        "repeticiones": repeticiones,
    }


def correlacion_equivalente(mi: float) -> float:
    """Traduce informacion mutua a la escala de la correlacion.

    Invierte la relacion que vale para una normal bivariada,
    ``MI = -0.5 * log(1 - rho^2)``. El resultado responde: que correlacion
    haria falta para generar esta cantidad de dependencia?

    Siempre es positivo: la informacion mutua mide **cuanta** dependencia hay,
    no en que direccion.
    """
    return float(np.sqrt(1 - np.exp(-2 * max(mi, 0.0))))


# ==========================================================================
# 2. Dependencia de cola
# ==========================================================================
def dependencia_de_cola(
    x: np.ndarray, y: np.ndarray, cuantil: float = CUANTIL_COLA,
    cola: str = "inferior",
) -> dict:
    """Probabilidad de que un activo sea extremo dado que el otro lo es.

    Parameters
    ----------
    cola
        ``"inferior"`` para caidas (el peor ``cuantil`` de cada serie) o
        ``"superior"`` para subidas.

    Returns
    -------
    dict con la probabilidad condicional, el numero de dias extremos conjuntos,
    y el ``exceso``: cuantas veces mas probable es la coincidencia de lo que
    seria bajo independencia. Un exceso de 1.0 significa independencia en la
    cola; 3.5 significa que las coincidencias son tres veces y media mas
    frecuentes que por azar.
    """
    if cola not in ("inferior", "superior"):
        raise ValueError("cola debe ser 'inferior' o 'superior'.")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bueno = np.isfinite(x) & np.isfinite(y)
    x, y = x[bueno], y[bueno]

    if cola == "inferior":
        ex = x <= np.quantile(x, cuantil)
        ey = y <= np.quantile(y, cuantil)
    else:
        ex = x >= np.quantile(x, 1 - cuantil)
        ey = y >= np.quantile(y, 1 - cuantil)

    n_x = int(ex.sum())
    conjuntos = int((ex & ey).sum())
    prob = conjuntos / n_x if n_x else np.nan

    return {
        "cola": cola,
        "probabilidad": prob,
        "dias_conjuntos": conjuntos,
        "dias_extremos": n_x,
        "exceso": prob / cuantil if n_x else np.nan,
    }


def asimetria_de_colas(
    x: np.ndarray, y: np.ndarray, cuantil: float = CUANTIL_COLA
) -> dict:
    """Compara la cola inferior contra la superior.

    Es la medida que la correlacion no puede dar: un solo coeficiente no puede
    expresar que las caidas van mas sincronizadas que las subidas. Si
    ``asimetria`` sale positiva, el par se desploma junto mas de lo que sube
    junto, y la diversificacion entre ambos falla justo en las crisis.
    """
    abajo = dependencia_de_cola(x, y, cuantil, "inferior")
    arriba = dependencia_de_cola(x, y, cuantil, "superior")
    return {
        "prob_caidas": abajo["probabilidad"],
        "prob_subidas": arriba["probabilidad"],
        "exceso_caidas": abajo["exceso"],
        "exceso_subidas": arriba["exceso"],
        "asimetria": abajo["probabilidad"] - arriba["probabilidad"],
    }


# ==========================================================================
# 3. Recorrido de muchos pares
# ==========================================================================
def tabla_no_lineal(
    returns: pd.DataFrame, bins: int = BINS, cuantil: float = CUANTIL_COLA,
    sectores: dict | None = None,
) -> pd.DataFrame:
    """Mide todos los pares de un universo y los ordena por brecha no lineal.

    Es la funcion que produce la evidencia: que pares tienen dependencia que la
    correlacion no capturo.

    Las series se discretizan una sola vez por activo y despues cada par solo
    requiere armar una tabla de 8x8, que es lo que hace viable recorrer miles de
    pares. Aun asi el costo crece con el cuadrado del numero de activos: con 50
    activos son 1,225 pares y toma segundos; con 500 serian 124,750.

    Returns
    -------
    DataFrame con una fila por par: correlacion de Pearson, informacion mutua,
    correlacion equivalente, brecha, y la dependencia en cada cola.
    """
    columnas = list(returns.columns)
    datos = returns.to_numpy(dtype=np.float64)
    n = len(datos)
    if n < 100:
        raise ValueError(f"Se necesitan al menos 100 observaciones, hay {n}.")

    # Precalculo por activo: casillas e indicadores de cola. Se hace una vez.
    indices = np.column_stack([_indices_por_cuantiles(datos[:, j], bins)
                               for j in range(len(columnas))])
    corte_bajo = np.quantile(datos, cuantil, axis=0)
    corte_alto = np.quantile(datos, 1 - cuantil, axis=0)
    es_bajo = datos <= corte_bajo
    es_alto = datos >= corte_alto

    pearson = np.corrcoef(datos, rowvar=False)

    filas = []
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            mi = _mi_desde_indices(indices[:, i], indices[:, j], bins, n, True)
            equivalente = correlacion_equivalente(mi)
            n_bajo = int(es_bajo[:, i].sum())
            n_alto = int(es_alto[:, i].sum())
            filas.append({
                "activo_a": columnas[i],
                "activo_b": columnas[j],
                "sector_a": (sectores or {}).get(columnas[i], ""),
                "sector_b": (sectores or {}).get(columnas[j], ""),
                "pearson": pearson[i, j],
                "info_mutua": mi,
                "corr_equivalente": equivalente,
                "brecha": equivalente - abs(pearson[i, j]),
                "prob_caidas": (es_bajo[:, i] & es_bajo[:, j]).sum() / n_bajo,
                "prob_subidas": (es_alto[:, i] & es_alto[:, j]).sum() / n_alto,
            })

    tabla = pd.DataFrame(filas)
    tabla["asimetria"] = tabla["prob_caidas"] - tabla["prob_subidas"]
    return tabla.sort_values("brecha", ascending=False).reset_index(drop=True)


def piso_de_brecha(
    returns: pd.DataFrame, bins: int = BINS, repeticiones: int = 300,
    semilla: int = 0,
) -> dict:
    """Cuanta brecha aparece entre series que **no** tienen relacion no lineal.

    Es la calibracion que vuelve interpretable la columna ``brecha``. El
    estimador de informacion mutua tiene dos errores que apuntan en direcciones
    opuestas: la correccion de sesgo deja un residuo positivo, y discretizar en
    casillas **pierde** informacion, lo que lo empuja hacia abajo. No hay forma
    de saber a priori cual gana, asi que se mide.

    El procedimiento revuelve una de las dos series de pares reales. Eso
    destruye cualquier relacion -lineal o no- conservando las distribuciones
    marginales, y la brecha que queda es puro artefacto del metodo. El percentil
    95 de esa distribucion es el umbral: por debajo de ese valor, una brecha no
    es evidencia de nada.

    Returns
    -------
    dict con la media, la desviacion y el percentil 95 de la brecha bajo
    independencia.
    """
    datos = returns.to_numpy(dtype=np.float64)
    n, k = datos.shape
    rng = np.random.default_rng(semilla)

    brechas = []
    for _ in range(repeticiones):
        i, j = rng.choice(k, 2, replace=False)
        x = datos[:, i]
        y = rng.permutation(datos[:, j])
        mi = _mi_desde_indices(
            _indices_por_cuantiles(x, bins), _indices_por_cuantiles(y, bins),
            bins, n, True,
        )
        brechas.append(correlacion_equivalente(mi) - abs(np.corrcoef(x, y)[0, 1]))

    brechas = np.array(brechas)
    return {
        "media": float(brechas.mean()),
        "desviacion": float(brechas.std()),
        "percentil_95": float(np.quantile(brechas, 0.95)),
        "maxima": float(brechas.max()),
        "repeticiones": repeticiones,
    }


def resumen_no_lineal(tabla: pd.DataFrame, piso: float | None = None) -> pd.DataFrame:
    """Resume la tabla de pares en las cifras que se reportan.

    Contesta de una sola vez: cuanta dependencia se le escapa a la correlacion
    en promedio, y hay asimetria entre caidas y subidas en el agregado?

    Parameters
    ----------
    piso
        Umbral de ``piso_de_brecha``. Si se pasa, el resumen incluye cuantos
        pares lo superan -- que son los unicos con no linealidad demostrable.
        Sin el, la brecha promedio no se puede interpretar.
    """
    filas = []
    if piso is not None:
        filas = [
            {"medida": "umbral de brecha (percentil 95 bajo independencia)", "valor": piso},
            {"medida": "pares que superan el umbral", "valor": float((tabla["brecha"] > piso).sum())},
            {"medida": "proporcion que lo supera", "valor": float((tabla["brecha"] > piso).mean())},
        ]
    return pd.DataFrame(filas + [
        {"medida": "pares medidos", "valor": len(tabla)},
        {"medida": "correlacion de Pearson promedio", "valor": tabla["pearson"].mean()},
        {"medida": "correlacion equivalente promedio", "valor": tabla["corr_equivalente"].mean()},
        {"medida": "brecha no lineal promedio", "valor": tabla["brecha"].mean()},
        {"medida": "brecha no lineal maxima", "valor": tabla["brecha"].max()},
        {"medida": "prob. de caer juntos (promedio)", "valor": tabla["prob_caidas"].mean()},
        {"medida": "prob. de subir juntos (promedio)", "valor": tabla["prob_subidas"].mean()},
        {"medida": "asimetria promedio (caidas - subidas)", "valor": tabla["asimetria"].mean()},
        {"medida": "pares con mas dependencia al caer", "valor": float((tabla["asimetria"] > 0).mean())},
    ])
