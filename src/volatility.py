"""Prediccion de volatilidad con GARCH(1,1).

El problema que resuelve
------------------------
El motor de riesgo del Entregable 2 reporta siempre el mismo numero: el
promedio de once anios. No cambia pase lo que pase en el mercado.

Pero el riesgo real si cambia. El notebook 06 lo midio: en los meses agitados
la volatilidad del mismo portafolio pasaba de 15.65% a 20.22%. Un motor que
reporta el promedio subestima el riesgo un 29% justo cuando alguien lo
consultaria.

La idea
-------
La volatilidad **se agrupa**: despues de un dia movido es mas probable que venga
otro dia movido. No se reparte al azar, viene en rachas. Es el mismo fenomeno
que el clima.

GARCH(1,1) convierte esa observacion en un modelo::

    varianza de maniana = omega + alpha * (golpe de hoy)^2 + beta * (varianza de hoy)

Tres ingredientes:

* ``omega`` fija el nivel al que la volatilidad regresa a la larga.
* ``alpha`` dice cuanto la mueve un golpe nuevo: la reaccion.
* ``beta`` dice cuanto arrastra del nivel anterior: la memoria.

``alpha + beta`` mide la persistencia. En mercados reales suele quedar entre
0.95 y 0.99: la volatilidad tarda semanas en volver a su nivel normal despues
de un susto. Si diera cerca de 0, la volatilidad de ayer no diria nada sobre la
de hoy y el modelo no serviria para predecir.

Por que esta implementado a mano
--------------------------------
El proyecto no depende de ninguna libreria de estadistica, y mantenerlo asi
tiene dos ventajas: cada linea es defendible, y el modelo no es una caja negra.
Es la misma decision que se tomo con el shrinkage de Ledoit-Wolf en
``src/risk.py``.

El ajuste usa **variance targeting**: en lugar de buscar los tres parametros a
la vez, ``omega`` se despeja de modo que la varianza de largo plazo del modelo
coincida con la varianza observada en los datos::

    omega = varianza_muestral * (1 - alpha - beta)

Eso deja solo dos parametros libres, y con dos basta una busqueda en
cuadricula: probar muchas combinaciones y quedarse con la que mejor explica los
datos. No es un parche, es una tecnica estandar -- ademas de reducir la
dimension, garantiza que el modelo reproduzca el nivel correcto de volatilidad
a largo plazo, cosa que la estimacion libre no asegura.
"""

from __future__ import annotations

import numpy as np

from src.risk import DIAS_HABILES

# Cota superior de la persistencia. Con alpha+beta >= 1 el modelo es explosivo:
# la varianza de largo plazo deja de existir y las predicciones divergen.
MAX_PERSISTENCIA = 0.999


def simular_varianzas(
    r: np.ndarray, omega: float, alpha: float, beta: float,
    var_inicial: float | None = None,
) -> np.ndarray:
    """Recorre la serie aplicando la recursion de GARCH(1,1).

    Devuelve la varianza que el modelo predecia para **cada** dia, usando solo
    informacion anterior a ese dia.

    Parameters
    ----------
    var_inicial
        Varianza del primer dia. Por defecto la varianza muestral, que es la
        estimacion mas razonable antes de ver nada -- pero es un estadistico de
        **muestra completa**, asi que la serie resultante cambia un poco si se
        recalcula con mas o menos dias. El efecto decae como ``beta^t`` y es
        irrelevante para describir el pasado. Para un uso donde la serie deba
        ser reproducible dia por dia en tiempo real conviene pasar un valor que
        no dependa de la muestra, por ejemplo la varianza de largo plazo que
        implican los propios parametros.
    """
    n = len(r)
    var = np.empty(n)
    var[0] = r.var() if var_inicial is None else var_inicial

    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]

    return var


def _log_verosimilitud(r: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """Que tan bien explican estos parametros los datos observados.

    Mide, para cada dia, que tan probable era el rendimiento que realmente
    ocurrio dada la varianza que el modelo predecia para ese dia. Un dia de -8%
    es muy improbable si el modelo esperaba calma, y normal si esperaba
    turbulencia. Sumando esa medida a lo largo de la serie se obtiene un
    puntaje: cuanto mas alto, mejor describen los parametros lo que paso.
    """
    var = simular_varianzas(r, omega, alpha, beta)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return -np.inf
    return float(-0.5 * np.sum(np.log(var) + r**2 / var))


def ajustar_garch(
    returns: np.ndarray, pasos_grilla: int = 40, refinamientos: int = 3
) -> dict:
    """Estima los parametros de GARCH(1,1) sobre una serie de rendimientos.

    Busca en cuadricula sobre ``alpha`` y ``beta``, refinando alrededor del
    mejor punto en cada pasada. ``omega`` no se busca: se despeja por variance
    targeting.

    Returns
    -------
    dict con omega, alpha, beta, la persistencia, la volatilidad de largo plazo
    anualizada y el puntaje de verosimilitud alcanzado.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        raise ValueError(f"Se necesitan al menos 100 observaciones, hay {len(r)}.")

    r = r - r.mean()  # GARCH modela las desviaciones, no el nivel
    var_objetivo = float(r.var())
    if var_objetivo <= 0:
        raise ValueError("La serie no tiene varianza; no hay nada que modelar.")

    # beta debe poder acercarse a cero: es la respuesta correcta cuando la serie
    # no tiene agrupamiento. Acotarlo por abajo obligaba al modelo a inventar
    # persistencia donde no la habia.
    lim_a, lim_b = (0.0, 0.30), (0.0, 0.998)
    mejor = {"puntaje": -np.inf}

    for pasada in range(refinamientos):
        for a in np.linspace(*lim_a, pasos_grilla):
            for b in np.linspace(*lim_b, pasos_grilla):
                if a + b >= MAX_PERSISTENCIA:
                    continue
                omega = var_objetivo * (1 - a - b)
                puntaje = _log_verosimilitud(r, omega, a, b)
                if puntaje > mejor["puntaje"]:
                    mejor = {"puntaje": puntaje, "alpha": a, "beta": b, "omega": omega}

        if mejor["puntaje"] == -np.inf:
            raise RuntimeError("Ningun conjunto de parametros resulto viable.")

        # Acercar la cuadricula alrededor del mejor punto encontrado.
        paso_a = (lim_a[1] - lim_a[0]) / pasos_grilla
        paso_b = (lim_b[1] - lim_b[0]) / pasos_grilla
        lim_a = (max(0.0, mejor["alpha"] - paso_a), min(0.5, mejor["alpha"] + paso_a))
        lim_b = (max(0.0, mejor["beta"] - paso_b), min(0.998, mejor["beta"] + paso_b))

    # Cuanto mejora el modelo frente a suponer volatilidad constante, que es
    # el caso alpha = beta = 0. Sin esta cifra, reportar la persistencia puede
    # enganiar: cuando alpha es casi cero el modelo ignora los golpes nuevos, y
    # entonces beta deja de estar identificado -- puede valer cualquier cosa sin
    # que el ajuste empeore. La persistencia solo significa algo si el modelo de
    # verdad le gana al de varianza constante.
    #
    # Como referencia: el estadistico 2*mejora se compara contra una chi
    # cuadrada de 2 grados de libertad, cuyo valor critico al 5% es 5.99. Una
    # mejora por debajo de 3 no es distinguible del azar.
    base = _log_verosimilitud(r, var_objetivo, 0.0, 0.0)
    mejora = mejor["puntaje"] - base

    persistencia = mejor["alpha"] + mejor["beta"]
    return {
        "omega": mejor["omega"],
        "alpha": mejor["alpha"],
        "beta": mejor["beta"],
        "persistencia": persistencia,
        "vol_largo_plazo": float(np.sqrt(var_objetivo * DIAS_HABILES)),
        "log_verosimilitud": mejor["puntaje"],
        "mejora_vs_constante": float(mejora),
        "hay_agrupamiento": bool(2 * mejora > 5.99),
        "n_obs": len(r),
    }


def predecir_volatilidad(
    returns: np.ndarray, parametros: dict, horizonte: int = 1
) -> np.ndarray:
    """Predice la volatilidad de los proximos ``horizonte`` dias.

    El primer dia sale directo de la recursion. A partir del segundo, como
    todavia no se conoce el golpe que va a ocurrir, se usa su valor esperado --
    que es justamente la varianza predicha. La formula colapsa entonces a::

        varianza(t+h) = largo_plazo + persistencia^(h-1) * (varianza(t+1) - largo_plazo)

    Es decir: la prediccion **converge hacia la volatilidad de largo plazo**, y
    la persistencia determina que tan rapido. Con persistencia 0.98 el regreso
    tarda meses; con 0.5, dias.

    Returns
    -------
    Volatilidades **anualizadas**, una por dia del horizonte.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    r = r - r.mean()

    omega, alpha, beta = parametros["omega"], parametros["alpha"], parametros["beta"]
    persistencia = alpha + beta

    var_hoy = simular_varianzas(r, omega, alpha, beta)[-1]
    var_maniana = omega + alpha * r[-1] ** 2 + beta * var_hoy

    var_largo_plazo = omega / (1 - persistencia) if persistencia < 1 else var_maniana

    pronostico = np.array([
        var_largo_plazo + persistencia ** h * (var_maniana - var_largo_plazo)
        for h in range(horizonte)
    ])
    return np.sqrt(np.maximum(pronostico, 0) * DIAS_HABILES)


def volatilidad_condicional(
    returns: np.ndarray, parametros: dict,
    media: float | None = None, var_inicial: float | None = None,
) -> np.ndarray:
    """Serie historica de la volatilidad que el modelo estimaba cada dia.

    Es lo que se grafica para mostrar que el modelo sigue las crisis: deberia
    dispararse en marzo de 2020 y bajar en los periodos tranquilos.

    Causalidad estricta
    -------------------
    Dados los parametros, la recursion solo mira hacia atras. Pero los dos
    valores con que arranca -la media que se resta y la varianza del primer
    dia- son por defecto estadisticos de **muestra completa**, asi que la serie
    no es exactamente la misma si se recalcula con mas dias. Medido sobre 3,000
    dias simulados, la diferencia reclasifica un 0.25% de los dias al aplicarle
    un umbral, y siempre dias que ya estaban pegados al umbral.

    Para describir el pasado eso es irrelevante. Para un uso que deba ser
    reproducible en tiempo real, pasar ``media`` y ``var_inicial`` explicitos
    elimina la dependencia por completo.

    Returns
    -------
    Volatilidades **anualizadas** (ya multiplicadas por raiz de 252), una por
    dia de la serie. No hay que volver a anualizarlas.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    r = r - (r.mean() if media is None else media)
    var = simular_varianzas(
        r, parametros["omega"], parametros["alpha"], parametros["beta"], var_inicial
    )
    return np.sqrt(var * DIAS_HABILES)
