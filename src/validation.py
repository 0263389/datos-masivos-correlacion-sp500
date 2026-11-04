"""Validacion hacia adelante (walk-forward).

Responde la pregunta que los Entregables 1 y 2 nunca se hicieron:

    Cuantos dias de historia conviene usar para estimar el riesgo?

Los dos entregables anteriores usaron **toda** la historia disponible sin
cuestionarlo. Este modulo mide si esa eleccion era la correcta.

El metodo
---------
La prueba del Entregable 2 partia la historia una sola vez: 70% para estimar,
30% para medir. Eso tiene dos problemas. El resultado depende de que le toco al
30% final -pudo salir bien o mal por suerte-, y no dice nada sobre cuanta
historia conviene usar.

La validacion hacia adelante repite el ejercicio avanzando en el tiempo::

    ventana de W dias -> arma portafolio -> lo mantiene 21 dias -> mide
    avanza 21 dias y repite

En cada paso el modelo **solo ve el pasado**: nunca usa informacion del periodo
que esta prediciendo. Por eso el resultado es una estimacion honesta de como se
comportaria en produccion.

Repitiendo todo con distintos valores de W aparece el punto optimo: la ventana
corta se adapta rapido pero tiene mucho ruido; la larga es estable pero arrastra
informacion que ya no aplica.

La restriccion que define el tamanio del universo
-------------------------------------------------
Con N activos, la covarianza muestral necesita mas de N observaciones para ser
invertible. Con 150 activos una ventana de 125 dias ni siquiera se puede usar.
El shrinkage de Ledoit-Wolf si lo permite, porque encoge hacia una matriz que
siempre es invertible -- y eso es, en si mismo, un resultado que vale la pena
medir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk import (
    DIAS_HABILES,
    equal_weights,
    ledoit_wolf_shrinkage,
    min_variance_weights,
    min_variance_weights_long_only,
    sample_covariance,
)

# Un mes de dias habiles. Rebalancear a diario no es realista -nadie reconstruye
# un portafolio cada dia- y hacerlo cada anio daria muy pocas pruebas.
REBALANCEO = 21


def _pesos(
    ventana: np.ndarray, metodo: str, encoger: bool, n_activos: int
) -> np.ndarray | None:
    """Construye el portafolio con los datos de una ventana.

    Devuelve ``None`` cuando la ventana no alcanza para estimar: menos
    observaciones que activos deja la covarianza muestral sin inversa.
    """
    if metodo == "iguales":
        return equal_weights(n_activos)

    if not encoger and len(ventana) <= n_activos:
        return None

    try:
        cov = (
            ledoit_wolf_shrinkage(ventana)[0]
            if encoger
            else sample_covariance(ventana)
        )
        if metodo == "min_var_largo":
            return min_variance_weights_long_only(cov)
        if metodo == "min_var_libre":
            return min_variance_weights(cov)
        raise ValueError(f"Metodo desconocido: {metodo}")
    except np.linalg.LinAlgError:
        # Covarianza singular: la ventana no da para este numero de activos.
        return None


def walk_forward(
    returns: pd.DataFrame,
    ventana: int,
    metodo: str = "min_var_largo",
    encoger: bool = True,
    rebalanceo: int = REBALANCEO,
    inicio_evaluacion: int | None = None,
) -> dict:
    """Corre la validacion hacia adelante para una ventana dada.

    Parameters
    ----------
    returns
        Rendimientos diarios, fechas en el indice, un activo por columna.
    ventana
        Cuantos dias de historia mira el modelo en cada rebalanceo.
    metodo
        ``min_var_largo``, ``min_var_libre`` o ``iguales``.
    encoger
        Si usa shrinkage de Ledoit-Wolf en lugar de la covarianza muestral.
    inicio_evaluacion
        Dia a partir del cual se empieza a medir. **Es el parametro que vuelve
        comparables dos ventanas distintas.**

        Sin el, cada ventana empieza a evaluar en cuanto tiene datos
        suficientes: una de 125 dias arranca en el dia 125 y una de 2000 en el
        dia 2000. Terminan midiendo periodos distintos -- la larga se salta el
        desplome de 2020 y la corta no -- y la comparacion deja de ser sobre
        cuanta historia mirar para ser sobre que periodo le toco a cada una.

        Fijandolo en el valor de la ventana mas larga del barrido, todas miden
        exactamente los mismos dias y la unica diferencia es cuanta historia
        usan para decidir.

    Returns
    -------
    dict con la volatilidad realizada, el numero de rebalanceos, la rotacion
    media de la cartera y la serie de rendimientos fuera de muestra.
    """
    X = returns.to_numpy(dtype=np.float64)
    T, N = X.shape

    if ventana + rebalanceo > T:
        raise ValueError(
            f"La ventana ({ventana}) mas un periodo de prueba ({rebalanceo}) "
            f"excede la historia disponible ({T} dias)."
        )

    primer_dia = ventana if inicio_evaluacion is None else max(ventana, inicio_evaluacion)
    if primer_dia + rebalanceo > T:
        raise ValueError(
            f"inicio_evaluacion={inicio_evaluacion} no deja periodo de prueba "
            f"con {T} dias de historia."
        )

    fuera_de_muestra: list[float] = []
    rotaciones: list[float] = []
    w_anterior: np.ndarray | None = None
    saltados = 0

    # Cada iteracion: mira hacia atras `ventana` dias, arma el portafolio, y lo
    # mantiene los siguientes `rebalanceo` dias sin volver a mirar los datos.
    for inicio in range(primer_dia, T - rebalanceo + 1, rebalanceo):
        w = _pesos(X[inicio - ventana : inicio], metodo, encoger, N)
        if w is None:
            saltados += 1
            continue

        fuera_de_muestra.extend(X[inicio : inicio + rebalanceo] @ w)

        if w_anterior is not None:
            # Rotacion: cuanto hay que mover de la cartera en cada rebalanceo.
            # Importa porque cada operacion cuesta comisiones en la practica.
            rotaciones.append(float(np.abs(w - w_anterior).sum() / 2))
        w_anterior = w

    if not fuera_de_muestra:
        raise ValueError(
            f"Ningun rebalanceo fue viable con ventana={ventana} y {N} activos. "
            "Con la covarianza muestral la ventana debe superar el numero de "
            "activos; prueba con encoger=True."
        )

    serie = np.asarray(fuera_de_muestra)
    return {
        "ventana": ventana,
        "inicio_evaluacion": primer_dia,
        "metodo": metodo,
        "encoger": encoger,
        "vol_realizada": float(serie.std(ddof=1) * np.sqrt(DIAS_HABILES)),
        "n_rebalanceos": len(rotaciones) + 1,
        "n_saltados": saltados,
        "dias_evaluados": len(serie),
        "rotacion_media": float(np.mean(rotaciones)) if rotaciones else 0.0,
        "serie": serie,
    }


def barrido_de_ventanas(
    returns: pd.DataFrame,
    ventanas: list[int],
    metodo: str = "min_var_largo",
    encoger: bool = True,
    rebalanceo: int = REBALANCEO,
    inicio_comun: bool = True,
) -> pd.DataFrame:
    """Repite la validacion para varias ventanas y devuelve la comparacion.

    La columna que importa es ``vol_realizada``: la volatilidad que el
    portafolio tuvo **de verdad** fuera de muestra. La ventana que la minimiza
    es la respuesta a la pregunta del modulo.

    **Por defecto todas las ventanas se evaluan sobre el mismo periodo**, el que
    impone la ventana mas larga. Sin eso la comparacion no mide lo que dice
    medir: cada ventana empezaria a evaluar en cuanto tiene datos, de modo que
    la mas larga se saltaria los primeros anios de historia. Si esos anios
    incluyen una crisis -2020 en estos datos- la ventana larga aparece como
    ganadora por haber sido evaluada en un periodo mas tranquilo, no por ser
    mejor.

    ``inicio_comun=False`` reproduce la version ingenua, util solo para mostrar
    la diferencia.

    ``periodos_sin_datos`` cuenta los rebalanceos que no se pudieron hacer
    porque la ventana era demasiado corta para el numero de activos. Una
    ventana con saltos no es comparable con una que corrio completa, y por eso
    se reporta en lugar de esconderse.
    """
    # El inicio comun se calcula solo sobre las ventanas VIABLES. Tomar el
    # maximo de la lista completa dejaria que una ventana imposible -mas larga
    # que la historia- arrastre a todas las demas y vacie el barrido.
    viables = [v for v in ventanas if v + rebalanceo <= len(returns)]
    inicio = max(viables) if (inicio_comun and viables) else None

    filas = []
    for v in ventanas:
        try:
            r = walk_forward(returns, v, metodo, encoger, rebalanceo, inicio)
        except ValueError:
            continue
        filas.append({
            "ventana_dias": r["ventana"],
            "ventana_anios": round(r["ventana"] / DIAS_HABILES, 1),
            "evalua_desde_dia": r["inicio_evaluacion"],
            "vol_realizada": r["vol_realizada"],
            "n_rebalanceos": r["n_rebalanceos"],
            "periodos_sin_datos": r["n_saltados"],
            "rotacion_media": r["rotacion_media"],
            "dias_evaluados": r["dias_evaluados"],
        })

    columnas = [
        "ventana_dias", "ventana_anios", "evalua_desde_dia", "vol_realizada",
        "n_rebalanceos", "periodos_sin_datos", "rotacion_media", "dias_evaluados",
    ]
    return pd.DataFrame(filas, columns=columnas)


def comparar_estrategias(
    returns: pd.DataFrame,
    ventana: int,
    rebalanceo: int = REBALANCEO,
) -> pd.DataFrame:
    """Compara los metodos de construccion sobre la misma ventana.

    Incluye ``iguales`` como punto de referencia: cualquier modelo de
    covarianza deberia superarlo, y con frecuencia no lo logra.
    """
    configs = [
        ("mín. varianza solo-largo (encogida)", "min_var_largo", True),
        ("mín. varianza solo-largo (muestral)", "min_var_largo", False),
        ("mín. varianza libre (encogida)", "min_var_libre", True),
        ("pesos iguales", "iguales", False),
    ]

    filas = []
    for etiqueta, metodo, encoger in configs:
        try:
            r = walk_forward(returns, ventana, metodo, encoger, rebalanceo)
        except ValueError:
            continue
        filas.append({
            "estrategia": etiqueta,
            "vol_realizada": r["vol_realizada"],
            "rotacion_media": r["rotacion_media"],
            "n_rebalanceos": r["n_rebalanceos"],
        })
    return pd.DataFrame(filas).sort_values("vol_realizada").reset_index(drop=True)
