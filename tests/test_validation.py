"""Pruebas de la validacion hacia adelante.

La prueba central es la de ausencia de look-ahead: verificar que el resultado
de los primeros periodos no cambia al alterar datos que ocurren despues. Si
cambiara, el modelo estaria usando informacion del futuro y toda la validacion
seria inutil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    REBALANCEO,
    barrido_de_ventanas,
    comparar_estrategias,
    walk_forward,
)


@pytest.fixture
def rendimientos() -> pd.DataFrame:
    """800 dias x 12 activos, modelo de un factor con ruido propio."""
    rng = np.random.default_rng(11)
    T, N = 800, 12
    factor = rng.standard_normal((T, 1))
    propio = rng.standard_normal((T, N))
    datos = 0.01 * (factor + propio)
    fechas = pd.bdate_range("2018-01-01", periods=T)
    return pd.DataFrame(datos, index=fechas, columns=[f"A{i:02d}" for i in range(N)])


class TestSinLookAhead:
    """La propiedad que vuelve valida toda la validacion."""

    def test_el_pasado_no_cambia_al_alterar_el_futuro(self, rendimientos):
        """Alterar la segunda mitad no debe mover los resultados de la primera.

        Se corre la validacion sobre la historia completa y sobre una copia en
        la que la segunda mitad fue reemplazada por ruido distinto. Los
        rendimientos fuera de muestra de la primera mitad tienen que ser
        identicos: si el modelo mirara hacia adelante, cambiarian.
        """
        ventana = 200
        original = walk_forward(rendimientos, ventana)["serie"]

        alterado = rendimientos.copy()
        corte = len(alterado) // 2
        rng = np.random.default_rng(999)
        alterado.iloc[corte:] = rng.standard_normal(alterado.iloc[corte:].shape) * 0.05
        nuevo = walk_forward(alterado, ventana)["serie"]

        # Dias fuera de muestra que terminan antes de que empiece la alteracion.
        comparables = corte - ventana
        assert comparables > 0, "la prueba necesita periodo limpio que comparar"
        assert np.allclose(original[:comparables], nuevo[:comparables])

    def test_la_alteracion_si_cambia_lo_posterior(self, rendimientos):
        """Control: si nada cambiara, la prueba anterior no probaria nada."""
        ventana = 200
        original = walk_forward(rendimientos, ventana)["serie"]

        alterado = rendimientos.copy()
        corte = len(alterado) // 2
        rng = np.random.default_rng(999)
        alterado.iloc[corte:] = rng.standard_normal(alterado.iloc[corte:].shape) * 0.05
        nuevo = walk_forward(alterado, ventana)["serie"]

        assert not np.allclose(original[-50:], nuevo[-50:])


class TestEstructura:
    def test_devuelve_las_claves_esperadas(self, rendimientos):
        r = walk_forward(rendimientos, 250)
        for clave in ["vol_realizada", "n_rebalanceos", "rotacion_media", "serie"]:
            assert clave in r

    def test_cuenta_bien_los_periodos(self, rendimientos):
        """Los dias evaluados deben ser exactamente rebalanceos x 21."""
        ventana = 250
        r = walk_forward(rendimientos, ventana)
        esperados = len(range(ventana, len(rendimientos) - REBALANCEO + 1, REBALANCEO))
        assert r["dias_evaluados"] == esperados * REBALANCEO

    def test_volatilidad_positiva_y_razonable(self, rendimientos):
        r = walk_forward(rendimientos, 250)
        assert 0 < r["vol_realizada"] < 2.0

    def test_rechaza_ventana_mas_larga_que_la_historia(self, rendimientos):
        with pytest.raises(ValueError, match="excede la historia"):
            walk_forward(rendimientos, len(rendimientos))


class TestRotacion:
    def test_pesos_iguales_no_rotan(self, rendimientos):
        """Si siempre repartes parejo, nunca mueves la cartera."""
        r = walk_forward(rendimientos, 250, metodo="iguales")
        assert r["rotacion_media"] == pytest.approx(0.0, abs=1e-12)

    def test_rotacion_entre_cero_y_uno(self, rendimientos):
        r = walk_forward(rendimientos, 250, metodo="min_var_largo")
        assert 0 <= r["rotacion_media"] <= 1


class TestVentanaCorta:
    """Con menos observaciones que activos, la muestral no alcanza."""

    def test_la_muestral_falla_con_ventana_corta(self, rendimientos):
        # 12 activos, ventana de 10 dias: no hay inversa posible.
        with pytest.raises(ValueError, match="Ningun rebalanceo fue viable"):
            walk_forward(rendimientos, 10, encoger=False)

    def test_el_shrinkage_si_lo_permite(self, rendimientos):
        r = walk_forward(rendimientos, 10, encoger=True)
        assert r["vol_realizada"] > 0
        assert r["n_saltados"] == 0


class TestBarrido:
    def test_una_fila_por_ventana_viable(self, rendimientos):
        tabla = barrido_de_ventanas(rendimientos, [200, 300, 400])
        assert len(tabla) == 3
        assert list(tabla["ventana_dias"]) == [200, 300, 400]

    def test_descarta_las_ventanas_imposibles(self, rendimientos):
        tabla = barrido_de_ventanas(rendimientos, [200, 10_000])
        assert list(tabla["ventana_dias"]) == [200]

    def test_comparar_estrategias_ordena_por_volatilidad(self, rendimientos):
        tabla = comparar_estrategias(rendimientos, 300)
        assert len(tabla) >= 2
        assert tabla["vol_realizada"].is_monotonic_increasing


class TestComparabilidadEntreVentanas:
    """El error que casi se cuela en el Entregable 3.

    Sin un inicio comun, cada ventana evalua un periodo distinto: la de 2000
    dias se salta los primeros ocho anios de historia y la de 125 no. Si esos
    anios contienen una crisis, la ventana larga gana por haber sido evaluada
    en un periodo mas tranquilo, no por predecir mejor.
    """

    def test_todas_evaluan_los_mismos_dias(self, rendimientos):
        tabla = barrido_de_ventanas(rendimientos, [100, 300, 500])
        assert tabla["dias_evaluados"].nunique() == 1, (
            "las ventanas no cubren el mismo periodo: la comparacion no es valida"
        )

    def test_todas_arrancan_el_mismo_dia(self, rendimientos):
        tabla = barrido_de_ventanas(rendimientos, [100, 300, 500])
        assert tabla["evalua_desde_dia"].nunique() == 1
        assert tabla["evalua_desde_dia"].iloc[0] == 500  # la ventana mas larga

    def test_sin_inicio_comun_no_son_comparables(self, rendimientos):
        """Control: demuestra que el problema existe de verdad."""
        tabla = barrido_de_ventanas(rendimientos, [100, 300, 500], inicio_comun=False)
        assert tabla["dias_evaluados"].nunique() > 1

    def test_el_inicio_comun_no_altera_la_ventana_mas_larga(self, rendimientos):
        """La mas larga ya empezaba ahi, asi que su resultado no debe cambiar."""
        con = walk_forward(rendimientos, 500, inicio_evaluacion=500)
        sin = walk_forward(rendimientos, 500)
        assert con["vol_realizada"] == pytest.approx(sin["vol_realizada"])
