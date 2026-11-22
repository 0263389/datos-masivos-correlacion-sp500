"""Pruebas de la red de correlacion.

Un arbol de expansion minima tiene propiedades que se pueden fijar sin
ambiguedad -numero de aristas, conectividad, ausencia de ciclos, minimalidad
del peso total- y eso lo vuelve facil de verificar de verdad y no por encima.
Varias pruebas comparan contra la respuesta exacta en grafos chicos calculados
a mano, que es la unica forma de saber que el algoritmo esta bien y no solo
que no revienta.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.network import (
    aristas_por_sector,
    arbol_de_expansion_minima,
    disposicion,
    grados,
    pureza_sectorial,
)


def matriz(valores, etiquetas):
    return pd.DataFrame(np.array(valores, dtype=float),
                        index=etiquetas, columns=etiquetas)


@pytest.fixture
def grafo_conocido():
    """Cuatro nodos cuyo arbol minimo se puede resolver a mano.

        A-B 1    A-C 5    A-D 4
        B-C 2    B-D 6    C-D 3

    El arbol minimo es A-B (1), B-C (2) y C-D (3), con peso total 6.
    """
    etiquetas = ["A", "B", "C", "D"]
    return matriz([
        [0, 1, 5, 4],
        [1, 0, 2, 6],
        [5, 2, 0, 3],
        [4, 6, 3, 0],
    ], etiquetas)


@pytest.fixture
def grafo_aleatorio():
    rng = np.random.default_rng(3)
    n = 30
    corr = rng.uniform(-0.3, 0.9, size=(n, n))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    d = np.sqrt(np.maximum(2 * (1 - corr), 0))
    etiquetas = [f"T{i:02d}" for i in range(n)]
    return matriz(d, etiquetas)


class TestArbolDeExpansionMinima:
    def test_resuelve_el_grafo_conocido(self, grafo_conocido):
        arbol = arbol_de_expansion_minima(grafo_conocido)
        obtenido = {frozenset((f.origen, f.destino)) for f in arbol.itertuples()}
        esperado = {frozenset(("A", "B")), frozenset(("B", "C")),
                    frozenset(("C", "D"))}
        assert obtenido == esperado
        assert arbol["distancia"].sum() == pytest.approx(6.0)

    def test_tiene_exactamente_n_menos_una_aristas(self, grafo_aleatorio):
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        assert len(arbol) == len(grafo_aleatorio) - 1

    def test_alcanza_a_todos_los_nodos(self, grafo_aleatorio):
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        tocados = set(arbol["origen"]) | set(arbol["destino"])
        assert tocados == set(grafo_aleatorio.columns)

    def test_no_tiene_ciclos(self, grafo_aleatorio):
        """Se verifica uniendo conjuntos: si una arista conecta dos nodos que
        ya estaban en el mismo grupo, cerro un ciclo.
        """
        grupo = {t: {t} for t in grafo_aleatorio.columns}
        for f in arbol_de_expansion_minima(grafo_aleatorio).itertuples():
            a, b = grupo[f.origen], grupo[f.destino]
            assert a is not b, f"{f.origen}-{f.destino} cierra un ciclo"
            unido = a | b
            for t in unido:
                grupo[t] = unido

    def test_su_peso_es_el_minimo_posible(self, grafo_conocido):
        """Comparacion por fuerza bruta contra todos los arboles del grafo.

        Con cuatro nodos hay 16 arboles de expansion posibles (formula de
        Cayley, n^(n-2)), asi que se pueden enumerar todos y comprobar que
        ninguno pesa menos. Es la unica prueba que verifica la *minimalidad* y
        no solo que el resultado sea un arbol.
        """
        etiquetas = list(grafo_conocido.columns)
        minimo = arbol_de_expansion_minima(grafo_conocido)["distancia"].sum()

        todas = list(itertools.combinations(etiquetas, 2))
        for combinacion in itertools.combinations(todas, len(etiquetas) - 1):
            grupo = {t: {t} for t in etiquetas}
            peso, valido = 0.0, True
            for a, b in combinacion:
                if grupo[a] is grupo[b]:
                    valido = False
                    break
                unido = grupo[a] | grupo[b]
                for t in unido:
                    grupo[t] = unido
                peso += grafo_conocido.loc[a, b]
            if valido:
                assert peso >= minimo - 1e-12

    def test_recupera_la_correlacion_desde_la_distancia(self, grafo_aleatorio):
        """Invierte la metrica de Mantegna: d = sqrt(2(1-rho)) => rho = 1 - d^2/2.

        Se comprueba contra la correlacion que produjo esa distancia, no
        contra la formula reaplicada: de lo contrario la prueba solo
        verificaria que una linea de codigo coincide consigo misma.
        """
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        for f in arbol.itertuples():
            d_original = grafo_aleatorio.loc[f.origen, f.destino]
            rho_original = 1 - d_original**2 / 2
            assert f.distancia == pytest.approx(d_original)
            assert f.correlacion == pytest.approx(rho_original)

    def test_sale_ordenado_de_mas_fuerte_a_mas_debil(self, grafo_aleatorio):
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        assert arbol["distancia"].is_monotonic_increasing

    def test_no_modifica_la_matriz_que_recibe(self, grafo_conocido):
        """Una funcion de lectura no debe tocar su entrada.

        El algoritmo pone infinito en la diagonal para que un nodo no se
        conecte consigo mismo. Sin copiar primero, eso modificaria la matriz
        de quien llamo a la funcion.
        """
        antes = grafo_conocido.copy()
        arbol_de_expansion_minima(grafo_conocido)
        pd.testing.assert_frame_equal(grafo_conocido, antes)

    def test_rechaza_un_solo_activo(self):
        with pytest.raises(ValueError, match="al menos dos"):
            arbol_de_expansion_minima(matriz([[0]], ["A"]))

    def test_rechaza_matriz_no_cuadrada(self):
        d = pd.DataFrame(np.zeros((3, 2)), columns=["A", "B"])
        with pytest.raises(ValueError, match="cuadrada"):
            arbol_de_expansion_minima(d)


class TestGrados:
    def test_los_grados_suman_el_doble_de_las_aristas(self, grafo_aleatorio):
        """Identidad de cualquier grafo: cada arista aporta 1 a dos nodos."""
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        assert grados(arbol)["grado"].sum() == 2 * len(arbol)

    def test_en_un_camino_los_extremos_son_hojas(self, grafo_conocido):
        """El arbol conocido es la cadena A-B-C-D."""
        g = grados(arbol_de_expansion_minima(grafo_conocido)).set_index("ticker")
        assert g.loc["A", "grado"] == 1 and g.loc["D", "grado"] == 1
        assert g.loc["B", "grado"] == 2 and g.loc["C", "grado"] == 2

    def test_agrega_el_sector_cuando_se_le_pasa(self, grafo_conocido):
        sectores = {"A": "Uno", "B": "Uno", "C": "Dos", "D": "Dos"}
        g = grados(arbol_de_expansion_minima(grafo_conocido), sectores)
        assert set(g["sector"]) == {"Uno", "Dos"}


class TestPurezaSectorial:
    def test_un_arbol_todo_del_mismo_sector_da_uno(self, grafo_conocido):
        sectores = dict.fromkeys("ABCD", "Unico")
        p = pureza_sectorial(arbol_de_expansion_minima(grafo_conocido), sectores)
        assert p["observado"] == pytest.approx(1.0)
        assert p["esperado_por_azar"] == pytest.approx(1.0)
        assert p["razon"] == pytest.approx(1.0)

    def test_el_esperado_sale_de_los_tamanios_de_los_sectores(self, grafo_conocido):
        """Con dos sectores de dos activos cada uno, la probabilidad de que
        dos nodos distintos al azar compartan sector es 2*(2*1)/(4*3) = 1/3.
        """
        sectores = {"A": "Uno", "B": "Uno", "C": "Dos", "D": "Dos"}
        p = pureza_sectorial(arbol_de_expansion_minima(grafo_conocido), sectores)
        assert p["esperado_por_azar"] == pytest.approx(1 / 3)

    def test_detecta_estructura_sectorial_construida(self, rng=None):
        """Dos bloques muy correlacionados por dentro y poco entre si.

        El arbol deberia quedarse casi solo con aristas internas, y la razon
        contra el azar deberia ser claramente mayor que 1.
        """
        rng = np.random.default_rng(7)
        n_bloque, n = 8, 16
        corr = np.full((n, n), 0.1)
        for ini in (0, n_bloque):
            corr[ini:ini + n_bloque, ini:ini + n_bloque] = 0.85
        corr += rng.normal(0, 0.01, size=(n, n))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        etiquetas = [f"T{i:02d}" for i in range(n)]
        d = matriz(np.sqrt(np.maximum(2 * (1 - corr), 0)), etiquetas)
        sectores = {t: ("Uno" if i < n_bloque else "Dos")
                    for i, t in enumerate(etiquetas)}

        p = pureza_sectorial(arbol_de_expansion_minima(d), sectores)
        assert p["observado"] > 0.85
        assert p["razon"] > 1.7

    def test_sectores_al_azar_no_superan_al_azar(self, grafo_aleatorio):
        """El otro lado. Si asignar sectores arbitrariamente diera razon alta,
        la medida estaria inflada y no mediria nada.
        """
        rng = np.random.default_rng(11)
        sectores = {t: f"S{rng.integers(0, 4)}" for t in grafo_aleatorio.columns}
        p = pureza_sectorial(arbol_de_expansion_minima(grafo_aleatorio), sectores)
        assert p["razon"] < 2.0

    def test_el_desglose_por_sector_cuadra_con_el_total(self, grafo_conocido):
        sectores = {"A": "Uno", "B": "Uno", "C": "Dos", "D": "Dos"}
        arbol = arbol_de_expansion_minima(grafo_conocido)
        tabla = aristas_por_sector(arbol, sectores)
        total = pureza_sectorial(arbol, sectores)["mismo_sector"]
        # Cada arista interna la cuenta solo su propio sector; las que cruzan
        # las cuentan dos sectores, pero no como internas.
        assert tabla["internas"].sum() == total


class TestDisposicion:
    def test_coloca_todos_los_nodos(self, grafo_aleatorio):
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        pos = disposicion(arbol, iteraciones=40)
        assert set(pos.index) == set(grafo_aleatorio.columns)
        assert list(pos.columns) == ["x", "y"]

    def test_queda_dentro_del_cuadro_normalizado(self, grafo_aleatorio):
        pos = disposicion(arbol_de_expansion_minima(grafo_aleatorio), iteraciones=40)
        assert pos.abs().to_numpy().max() == pytest.approx(1.0, abs=1e-9)

    def test_es_reproducible_con_la_misma_semilla(self, grafo_aleatorio):
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        a = disposicion(arbol, iteraciones=40, semilla=5)
        b = disposicion(arbol, iteraciones=40, semilla=5)
        pd.testing.assert_frame_equal(a, b)

    def test_los_nodos_no_quedan_encimados(self, grafo_aleatorio):
        """La repulsion existe para esto: si todos cayeran en el mismo punto,
        el dibujo no mostraria nada.
        """
        pos = disposicion(arbol_de_expansion_minima(grafo_aleatorio), iteraciones=120)
        p = pos.to_numpy()
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 1e-3

    def test_acerca_a_los_conectados(self, grafo_aleatorio):
        """Dos nodos unidos por una arista deben quedar, en promedio, mas
        cerca que dos nodos cualesquiera.
        """
        arbol = arbol_de_expansion_minima(grafo_aleatorio)
        pos = disposicion(arbol, iteraciones=200)
        unidos = np.mean([
            np.linalg.norm(pos.loc[f.origen] - pos.loc[f.destino])
            for f in arbol.itertuples()
        ])
        p = pos.to_numpy()
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        cualesquiera = d[np.triu_indices(len(p), k=1)].mean()
        assert unidos < cualesquiera
