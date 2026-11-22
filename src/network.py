"""La red de correlacion del mercado: su esqueleto y sus concentradores.

El problema que resuelve
------------------------
Con 458 activos hay 104,653 correlaciones. La matriz las contiene todas y por
eso no se puede leer: cada accion esta conectada con todas las demas, de modo
que no hay estructura visible, solo una nube de numeros.

El arbol de expansion minima resuelve eso quedandose con **la minima cantidad
de conexiones que mantiene unido a todo el mercado**: exactamente n - 1
aristas, las mas cortas posibles, sin ciclos. Pasa de 104,653 relaciones a 457,
y las que sobreviven son las que de verdad sostienen la estructura.

La tecnica es de Mantegna (1999), *Hierarchical structure in financial
markets*, el mismo trabajo del que sale la metrica de distancia que ya usa
``analysis.distance_matrix``.

Que responde que la correlacion promedio no responde
-----------------------------------------------------
El Entregable 1 pregunto si las acciones del mismo sector se parecen **en
promedio**. Esta es una version mas exigente de la misma pregunta:

    Cuando te quedas solo con las conexiones mas fuertes del mercado,
    cuantas caen dentro del mismo sector?

Un promedio alto puede salir de muchas relaciones tibias. Si el sector de
verdad organiza el mercado, deberia dominar tambien **el esqueleto**, que es
donde solo caben las relaciones mas fuertes.

La comparacion necesita una referencia, y no es cero. Si las aristas se
repartieran al azar, una fraccion de ellas caeria dentro del mismo sector solo
porque los sectores grandes tienen mas parejas posibles. Esa referencia se
calcula y se reporta junto al dato observado: sin ella, un 30% no significa
nada.

Por que esta implementado a mano
--------------------------------
El proyecto no depende de librerias de grafos, por la misma razon que no
depende de librerias de estadistica: cada linea es defendible. El algoritmo de
Prim son quince lineas de numpy, y la disposicion de los nodos para dibujarlos
es el metodo de fuerzas de Fruchterman y Reingold (1991), que son otras veinte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ==========================================================================
# 1. El esqueleto
# ==========================================================================
def arbol_de_expansion_minima(distancias: pd.DataFrame) -> pd.DataFrame:
    """Las n - 1 conexiones mas cortas que mantienen unido a todo el mercado.

    Algoritmo de Prim: arranca en un nodo y, en cada paso, agrega la arista
    mas corta que conecta el arbol con algo que todavia no esta dentro. El
    resultado no depende del nodo inicial mientras las distancias no tengan
    empates exactos, cosa que con correlaciones de datos reales no ocurre.

    El costo es cuadratico en el numero de activos, que para 458 es
    inmediato. La version con monticulo seria mejor para grafos dispersos, y
    este no lo es: todos los pares tienen distancia.

    Parameters
    ----------
    distancias
        Matriz cuadrada y simetrica, normalmente de
        ``analysis.distance_matrix``.

    Returns
    -------
    DataFrame con una fila por arista: los dos extremos, la distancia y la
    correlacion que le corresponde, ordenadas de la conexion mas fuerte a la
    mas debil.
    """
    etiquetas = list(distancias.columns)
    n = len(etiquetas)
    if n < 2:
        raise ValueError(f"Hacen falta al menos dos activos, hay {n}.")
    if distancias.shape[0] != distancias.shape[1]:
        raise ValueError(f"La matriz debe ser cuadrada, es {distancias.shape}.")

    # La copia no es opcional. ``to_numpy`` devuelve una vista de solo lectura
    # sobre los datos del DataFrame, asi que sin copiar o bien revienta, o bien
    # -si la vista fuera escribible- ``fill_diagonal`` modificaria la matriz de
    # quien llamo a la funcion. Una funcion de lectura no debe tocar su entrada.
    D = distancias.to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(D, np.inf)  # un nodo no se conecta consigo mismo

    en_arbol = np.zeros(n, dtype=bool)
    en_arbol[0] = True
    # Para cada nodo fuera del arbol: a que distancia esta y de quien.
    distancia_al_arbol = D[0].copy()
    vecino_en_arbol = np.zeros(n, dtype=int)

    filas = []
    for _ in range(n - 1):
        candidatos = np.where(en_arbol, np.inf, distancia_al_arbol)
        j = int(np.argmin(candidatos))
        filas.append({
            "origen": etiquetas[vecino_en_arbol[j]],
            "destino": etiquetas[j],
            "distancia": float(distancia_al_arbol[j]),
        })
        en_arbol[j] = True

        # El nodo recien agregado puede acercar a los que siguen afuera.
        mas_cerca = D[j] < distancia_al_arbol
        vecino_en_arbol[mas_cerca & ~en_arbol] = j
        distancia_al_arbol = np.minimum(distancia_al_arbol, D[j])

    arbol = pd.DataFrame(filas)
    # Se invierte la metrica de Mantegna: d = sqrt(2(1-rho)) => rho = 1 - d^2/2
    arbol["correlacion"] = 1.0 - arbol["distancia"] ** 2 / 2.0
    return arbol.sort_values("distancia").reset_index(drop=True)


# ==========================================================================
# 2. Quien sostiene la red
# ==========================================================================
def grados(aristas: pd.DataFrame, sectores: dict | None = None) -> pd.DataFrame:
    """Cuantas conexiones del esqueleto toca cada activo.

    En un arbol la suma de los grados es exactamente ``2 * (n - 1)``, asi que
    el promedio ronda 2 y casi todos los nodos son hojas. Los pocos que se
    salen de ahi son **concentradores**: activos que el mercado usa como punto
    de paso para conectar partes distintas de si mismo.

    Importa para el riesgo. Un concentrador no es necesariamente el activo mas
    volatil ni el mas correlacionado con el indice; es el que, si se mueve,
    arrastra a mas vecinos directos.
    """
    conteo = pd.concat([aristas["origen"], aristas["destino"]]).value_counts()
    tabla = conteo.rename_axis("ticker").reset_index(name="grado")
    if sectores:
        tabla["sector"] = tabla["ticker"].map(sectores).fillna("")
    return tabla


def pureza_sectorial(aristas: pd.DataFrame, sectores: dict) -> dict:
    """Que fraccion del esqueleto conecta activos del mismo sector.

    El dato observado por si solo no dice nada, porque algunas aristas caerian
    dentro del mismo sector aunque se repartieran al azar: los sectores grandes
    tienen mas parejas posibles. La referencia correcta es la probabilidad de
    que dos nodos distintos elegidos al azar compartan sector::

        esperado = suma_s [ n_s * (n_s - 1) ] / [ n * (n - 1) ]

    donde ``n_s`` es el numero de activos del sector ``s``. Se reporta junto al
    observado y como razon entre ambos.
    """
    nodos = pd.concat([aristas["origen"], aristas["destino"]]).unique()
    por_sector = pd.Series([sectores.get(x, "") for x in nodos]).value_counts()
    n = int(por_sector.sum())
    if n < 2:
        raise ValueError("Hacen falta al menos dos activos con sector asignado.")

    esperado = float((por_sector * (por_sector - 1)).sum() / (n * (n - 1)))

    mismo = aristas.apply(
        lambda f: sectores.get(f["origen"], "") == sectores.get(f["destino"], ""),
        axis=1,
    )
    observado = float(mismo.mean())

    return {
        "aristas": len(aristas),
        "mismo_sector": int(mismo.sum()),
        "observado": observado,
        "esperado_por_azar": esperado,
        "razon": observado / esperado if esperado else np.nan,
        "sectores": int(len(por_sector)),
    }


def aristas_por_sector(aristas: pd.DataFrame, sectores: dict) -> pd.DataFrame:
    """Desglosa por sector cuantas de sus aristas se quedan dentro.

    Un sector con proporcion alta forma un bloque compacto y aislado; uno con
    proporcion baja esta repartido por la red y se conecta sobre todo hacia
    afuera.
    """
    filas = []
    for sector in sorted({s for s in sectores.values() if s}):
        toca = aristas.apply(
            lambda f: sectores.get(f["origen"]) == sector
            or sectores.get(f["destino"]) == sector,
            axis=1,
        )
        if not toca.any():
            continue
        sub = aristas[toca]
        dentro = sub.apply(
            lambda f: sectores.get(f["origen"]) == sectores.get(f["destino"]),
            axis=1,
        )
        filas.append({
            "sector": sector,
            "aristas": int(toca.sum()),
            "internas": int(dentro.sum()),
            "proporcion_interna": float(dentro.mean()),
            "distancia_media": float(sub["distancia"].mean()),
        })
    return pd.DataFrame(filas).sort_values(
        "proporcion_interna", ascending=False).reset_index(drop=True)


# ==========================================================================
# 3. Disposicion para dibujar
# ==========================================================================
def disposicion(
    aristas: pd.DataFrame, iteraciones: int = 220, semilla: int = 0
) -> pd.DataFrame:
    """Coloca los nodos en el plano con el metodo de fuerzas.

    Fruchterman y Reingold (1991). El grafo se trata como un sistema fisico:
    todos los nodos se repelen entre si, y los que comparte una arista se
    atraen. Al dejarlo relajar, los grupos muy conectados terminan juntos sin
    que nadie le diga al algoritmo que existen los sectores -- y por eso, si
    los sectores aparecen agrupados en el dibujo, es un hallazgo y no una
    decision de presentacion.

    La temperatura baja en cada iteracion para que los nodos dejen de moverse y
    el resultado converja en lugar de oscilar.

    Returns
    -------
    DataFrame indexado por ticker con las columnas ``x`` e ``y``, normalizadas
    al cuadrado [-1, 1].
    """
    nodos = sorted(set(aristas["origen"]) | set(aristas["destino"]))
    indice = {t: i for i, t in enumerate(nodos)}
    n = len(nodos)

    enlaces = np.array([[indice[a], indice[b]]
                        for a, b in zip(aristas["origen"], aristas["destino"])])

    rng = np.random.default_rng(semilla)
    pos = rng.uniform(-1, 1, size=(n, 2))

    k = np.sqrt(1.0 / n)          # distancia de equilibrio entre nodos
    temperatura = 0.1
    enfriamiento = temperatura / (iteraciones + 1)

    for _ in range(iteraciones):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)          # sin auto-repulsion
        dist = np.maximum(dist, 1e-9)

        # Repulsion de todos contra todos: k^2 / d
        fuerza = (delta / dist[..., None]) * (k**2 / dist)[..., None]
        desplazamiento = fuerza.sum(axis=1)

        # Atraccion solo entre los conectados: d^2 / k
        a, b = enlaces[:, 0], enlaces[:, 1]
        d_ab = pos[a] - pos[b]
        norma = np.maximum(np.linalg.norm(d_ab, axis=1), 1e-9)
        tira = (d_ab / norma[:, None]) * (norma**2 / k)[:, None]
        np.add.at(desplazamiento, a, -tira)
        np.add.at(desplazamiento, b, tira)

        # Avanzar sin pasarse de la temperatura actual.
        paso = np.maximum(np.linalg.norm(desplazamiento, axis=1), 1e-9)
        pos += (desplazamiento / paso[:, None]) * np.minimum(paso, temperatura)[:, None]
        temperatura -= enfriamiento

    pos -= pos.mean(axis=0)
    escala = np.abs(pos).max() or 1.0
    return pd.DataFrame(pos / escala, index=nodos, columns=["x", "y"])
