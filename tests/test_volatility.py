"""Pruebas del modelo GARCH(1,1).

La prueba central es la de recuperacion: se simulan datos con parametros
conocidos y se verifica que el estimador los encuentre. Es la unica forma de
saber que el ajuste funciona, porque sobre datos reales no existe una respuesta
correcta contra la cual comparar.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.risk import DIAS_HABILES
from src.volatility import (
    ajustar_garch,
    predecir_volatilidad,
    simular_varianzas,
    volatilidad_condicional,
)


def generar_garch(
    n: int, omega: float, alpha: float, beta: float, semilla: int = 0
) -> np.ndarray:
    """Genera una serie que de verdad sigue un GARCH(1,1) con esos parametros."""
    rng = np.random.default_rng(semilla)
    r = np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


class TestRecuperacionDeParametros:
    """Si el estimador no recupera parametros conocidos, no sirve."""

    @pytest.mark.parametrize(
        "alpha,beta",
        [(0.08, 0.90), (0.05, 0.93), (0.15, 0.80)],
    )
    def test_recupera_alpha_y_beta(self, alpha, beta):
        omega = 1e-5 * (1 - alpha - beta)
        datos = generar_garch(6000, omega, alpha, beta, semilla=7)
        r = ajustar_garch(datos)

        # La tolerancia es amplia a proposito: con 6000 observaciones el error
        # de estimacion de GARCH es real, no un defecto del metodo.
        assert abs(r["alpha"] - alpha) < 0.06
        assert abs(r["beta"] - beta) < 0.10

    def test_recupera_la_persistencia(self):
        """La persistencia se estima mejor que alpha y beta por separado.

        Es lo esperable: alpha y beta se compensan entre si, pero su suma
        -- que es lo que determina cuanto dura un susto -- queda bien anclada.
        """
        alpha, beta = 0.08, 0.90
        datos = generar_garch(6000, 1e-5 * (1 - alpha - beta), alpha, beta, semilla=3)
        r = ajustar_garch(datos)
        assert abs(r["persistencia"] - (alpha + beta)) < 0.04

    def test_detecta_ausencia_de_agrupamiento(self):
        """Con ruido sin estructura, el modelo debe reportar que no hay rachas.

        No se verifica la persistencia sino el diagnostico, y la razon es
        estadistica: cuando alpha es casi cero el modelo ignora los golpes
        nuevos, y entonces beta deja de estar identificado -- puede tomar
        cualquier valor sin que el ajuste empeore. Sobre esta serie la
        diferencia de verosimilitud entre beta=0.001 y beta=0.99 es de 0.44
        sobre 4000 observaciones: ruido.

        Lo que si distingue los dos casos es si el modelo le gana al de
        varianza constante. Sobre ruido puro, no debe ganarle.
        """
        rng = np.random.default_rng(5)
        r = ajustar_garch(rng.standard_normal(4000) * 0.01)
        assert not r["hay_agrupamiento"], (
            f"reporto agrupamiento sobre ruido puro "
            f"(mejora={r['mejora_vs_constante']:.2f})"
        )
        assert r["alpha"] < 0.02

    def test_si_detecta_agrupamiento_cuando_lo_hay(self):
        """Control: sobre una serie con GARCH real, debe detectarlo."""
        datos = generar_garch(4000, 1e-5 * 0.02, 0.08, 0.90, semilla=5)
        r = ajustar_garch(datos)
        assert r["hay_agrupamiento"]
        assert r["mejora_vs_constante"] > 50


class TestPropiedades:
    @pytest.fixture
    def serie(self):
        return generar_garch(3000, 1e-5 * 0.02, 0.08, 0.90, semilla=11)

    def test_parametros_en_rango_valido(self, serie):
        r = ajustar_garch(serie)
        assert r["omega"] > 0
        assert 0 < r["alpha"] < 1
        assert 0 < r["beta"] < 1
        assert r["persistencia"] < 1, "persistencia >= 1 seria un modelo explosivo"

    def test_variance_targeting_reproduce_la_volatilidad_observada(self, serie):
        """Es la propiedad que garantiza el metodo de ajuste."""
        r = ajustar_garch(serie)
        esperada = float(np.std(serie - serie.mean()) * np.sqrt(DIAS_HABILES))
        assert r["vol_largo_plazo"] == pytest.approx(esperada, rel=1e-6)

    def test_varianzas_positivas_y_finitas(self, serie):
        r = ajustar_garch(serie)
        var = simular_varianzas(serie - serie.mean(), r["omega"], r["alpha"], r["beta"])
        assert np.all(var > 0)
        assert np.all(np.isfinite(var))

    def test_rechaza_serie_demasiado_corta(self):
        with pytest.raises(ValueError, match="al menos 100"):
            ajustar_garch(np.zeros(50))

    def test_rechaza_serie_sin_varianza(self):
        with pytest.raises(ValueError, match="no tiene varianza"):
            ajustar_garch(np.ones(500))


class TestPrediccion:
    @pytest.fixture
    def ajuste(self):
        serie = generar_garch(3000, 1e-5 * 0.02, 0.08, 0.90, semilla=13)
        return serie, ajustar_garch(serie)

    def test_devuelve_una_volatilidad_por_dia(self, ajuste):
        serie, par = ajuste
        assert len(predecir_volatilidad(serie, par, horizonte=10)) == 10

    def test_todas_positivas(self, ajuste):
        serie, par = ajuste
        assert np.all(predecir_volatilidad(serie, par, horizonte=30) > 0)

    def test_converge_hacia_el_largo_plazo(self, ajuste):
        """A horizonte largo la prediccion debe olvidar lo que pasa hoy."""
        serie, par = ajuste
        lejos = predecir_volatilidad(serie, par, horizonte=500)[-1]
        assert lejos == pytest.approx(par["vol_largo_plazo"], rel=0.05)

    def test_reacciona_a_un_golpe_reciente(self, ajuste):
        """Si el ultimo dia fue violento, la prediccion debe subir."""
        serie, par = ajuste
        tranquilo = predecir_volatilidad(serie, par, horizonte=1)[0]

        agitado = serie.copy()
        agitado[-1] = 10 * serie.std()  # un dia de shock
        con_golpe = predecir_volatilidad(agitado, par, horizonte=1)[0]

        assert con_golpe > tranquilo * 1.2


class TestVolatilidadCondicional:
    def test_sigue_los_periodos_agitados(self):
        """La serie estimada debe subir donde los datos son mas violentos."""
        rng = np.random.default_rng(17)
        tranquilo = rng.standard_normal(1000) * 0.005
        agitado = rng.standard_normal(300) * 0.04
        serie = np.concatenate([tranquilo, agitado, tranquilo])

        par = ajustar_garch(serie)
        vol = volatilidad_condicional(serie, par)

        assert vol[1100:1250].mean() > 2 * vol[200:800].mean()

    def test_longitud_y_signo(self):
        serie = generar_garch(1500, 1e-5 * 0.02, 0.08, 0.90, semilla=19)
        par = ajustar_garch(serie)
        vol = volatilidad_condicional(serie, par)
        assert len(vol) == len(serie)
        assert np.all(vol > 0)
