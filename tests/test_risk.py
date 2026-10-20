"""Pruebas del motor de riesgo.

Se ejecutan sin Spark y sin conexion a internet: usan datos sinteticos con
estructura conocida, de modo que la respuesta correcta se sabe de antemano.

Correr con:

    ./.venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    DIAS_HABILES,
    classify_regimes,
    condition_number,
    equal_weights,
    ledoit_wolf_shrinkage,
    min_variance_weights,
    regime_comparison,
    risk_decomposition,
    sample_covariance,
)


@pytest.fixture
def rendimientos_sinteticos() -> np.ndarray:
    """500 dias x 30 activos con correlacion conocida de 0.4.

    Se construyen como ``factor_comun + ruido_propio``, que es exactamente el
    modelo de un factor del Entregable 2. Con pesos iguales en ambos terminos
    la correlacion teorica entre cualquier par es 0.5.
    """
    rng = np.random.default_rng(7)
    T, N = 500, 30
    factor = rng.standard_normal((T, 1))
    propio = rng.standard_normal((T, N))
    return 0.01 * (factor + propio)


class TestShrinkage:
    def test_delta_en_intervalo_valido(self, rendimientos_sinteticos):
        _, delta, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos)
        assert 0.0 <= delta <= 1.0

    def test_matriz_simetrica(self, rendimientos_sinteticos):
        cov, _, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos)
        assert np.allclose(cov, cov.T)

    def test_semidefinida_positiva(self, rendimientos_sinteticos):
        """Una covarianza real no puede tener valores propios negativos."""
        cov, _, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos)
        assert np.linalg.eigvalsh(cov).min() > 0

    def test_conserva_las_varianzas(self, rendimientos_sinteticos):
        """El objetivo de correlacion constante solo encoge la correlacion.

        La diagonal -la volatilidad individual de cada activo- debe quedar
        intacta: es informacion que se estima con T datos, no con T/N.
        """
        cov, _, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos, annualize=False)
        Y = rendimientos_sinteticos - rendimientos_sinteticos.mean(axis=0)
        S = (Y.T @ Y) / len(Y)
        assert np.allclose(np.diag(cov), np.diag(S))

    def test_mejora_el_condicionamiento(self):
        """Con pocos datos frente a activos, el shrinkage debe ayudar."""
        rng = np.random.default_rng(11)
        X = 0.01 * (rng.standard_normal((80, 1)) + rng.standard_normal((80, 60)))
        encogida, delta, _ = ledoit_wolf_shrinkage(X)
        assert delta > 0
        assert condition_number(encogida) < condition_number(sample_covariance(X))

    def test_anualizacion(self, rendimientos_sinteticos):
        diaria, _, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos, annualize=False)
        anual, _, _ = ledoit_wolf_shrinkage(rendimientos_sinteticos, annualize=True)
        assert np.allclose(anual, diaria * DIAS_HABILES)

    def test_rechaza_varianza_cero(self):
        X = np.zeros((100, 5))
        with pytest.raises(ValueError, match="varianza cero"):
            ledoit_wolf_shrinkage(X)


class TestDescomposicionDelRiesgo:
    def test_las_contribuciones_suman_la_volatilidad(self, rendimientos_sinteticos):
        """La identidad central: el reparto del riesgo es exacto."""
        cov = sample_covariance(rendimientos_sinteticos)
        r = risk_decomposition(cov)
        assert r["check_sum"] < 1e-12
        assert np.isclose(r["pct_risk"].sum(), 1.0)

    def test_activos_independientes_dan_maxima_diversificacion(self):
        """Sin correlacion, el ratio debe alcanzar sqrt(n)."""
        n = 25
        cov = np.eye(n) * 0.04
        r = risk_decomposition(cov)
        assert np.isclose(r["diversification_ratio"], np.sqrt(n), rtol=1e-9)

    def test_activos_identicos_no_diversifican(self):
        """Con correlacion 1 el ratio cae a 1: se comportan como una posicion."""
        n = 10
        cov = np.full((n, n), 0.04)
        r = risk_decomposition(cov)
        assert np.isclose(r["diversification_ratio"], 1.0, rtol=1e-9)

    def test_el_riesgo_no_es_proporcional_al_peso(self):
        """Dos activos sin correlacion, uno el triple de volatil.

        Con pesos iguales, el volatil debe aportar mucho mas de la mitad del
        riesgo. Es el hallazgo que hace util la descomposicion.
        """
        cov = np.diag([0.01, 0.09])  # volatilidades 10% y 30%
        r = risk_decomposition(cov, labels=["tranquilo", "volatil"])
        assert r["pct_risk"][1] > 0.85
        assert r["weights"][1] == 0.5

    def test_normaliza_los_pesos(self):
        cov = np.eye(3) * 0.04
        r = risk_decomposition(cov, weights=[2, 2, 2])
        assert np.allclose(r["weights"], 1 / 3)

    def test_rechaza_pesos_de_tamanio_equivocado(self):
        with pytest.raises(ValueError, match="no coinciden"):
            risk_decomposition(np.eye(3), weights=[0.5, 0.5])


class TestPortafolios:
    def test_minima_varianza_mejora_a_pesos_iguales_dentro_de_muestra(
        self, rendimientos_sinteticos
    ):
        cov = sample_covariance(rendimientos_sinteticos)
        w_min = min_variance_weights(cov)
        w_igual = equal_weights(cov.shape[0])
        assert (w_min @ cov @ w_min) <= (w_igual @ cov @ w_igual)

    def test_los_pesos_suman_uno(self, rendimientos_sinteticos):
        w = min_variance_weights(sample_covariance(rendimientos_sinteticos))
        assert np.isclose(w.sum(), 1.0)

    def test_minima_varianza_carga_el_activo_tranquilo(self):
        """Sin correlacion, el peso debe ser inverso a la varianza."""
        cov = np.diag([0.01, 0.04])
        w = min_variance_weights(cov)
        assert w[0] > w[1]
        assert np.isclose(w[0] / w[1], 4.0, rtol=1e-9)


class TestRegimenes:
    def test_los_tercios_quedan_balanceados(self):
        rng = np.random.default_rng(3)
        serie = pd.Series(rng.standard_normal(90))
        conteos = classify_regimes(serie).value_counts()
        assert conteos.min() >= 25  # 90/3 = 30, con holgura por empates

    def test_los_meses_malos_son_los_peores(self):
        serie = pd.Series([-0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
        regimenes = classify_regimes(serie)
        assert regimenes.iloc[0] == "meses malos"
        assert regimenes.iloc[-1] == "meses buenos"

    def test_comparacion_por_regimen_devuelve_todas_las_filas(self):
        rng = np.random.default_rng(5)
        fechas = pd.bdate_range("2020-01-01", periods=400)
        datos = pd.DataFrame(
            0.01 * rng.standard_normal((len(fechas), 4)),
            index=fechas, columns=["A", "B", "C", "D"],
        )
        mensual = datos.mean(axis=1).groupby(datos.index.to_period("M")).sum()
        tabla = regime_comparison(datos, classify_regimes(mensual), ["A", "B", "C", "D"])

        assert "todo el periodo" in tabla["regimen"].values
        assert len(tabla) == 4
        assert (tabla["ratio_diversificacion"] > 0).all()
        assert (tabla["vol_portafolio"] > 0).all()


class TestLargoSolamente:
    """Minima varianza con restriccion de no vender en corto."""

    def test_sin_posiciones_en_corto(self, rendimientos_sinteticos):
        from src.risk import min_variance_weights_long_only
        w = min_variance_weights_long_only(sample_covariance(rendimientos_sinteticos))
        assert (w >= -1e-9).all()

    def test_los_pesos_suman_uno(self, rendimientos_sinteticos):
        from src.risk import min_variance_weights_long_only
        w = min_variance_weights_long_only(sample_covariance(rendimientos_sinteticos))
        assert np.isclose(w.sum(), 1.0)

    def test_nunca_mejora_a_la_version_sin_restriccion(self, rendimientos_sinteticos):
        """Restringir no puede bajar la varianza: el optimo libre es una cota."""
        from src.risk import min_variance_weights_long_only
        cov = sample_covariance(rendimientos_sinteticos)
        w_libre = min_variance_weights(cov)
        w_largo = min_variance_weights_long_only(cov)
        assert (w_largo @ cov @ w_largo) >= (w_libre @ cov @ w_libre) - 1e-12

    def test_mejora_a_pesos_iguales(self, rendimientos_sinteticos):
        """Pero si debe superar al punto de comparacion ingenuo."""
        from src.risk import min_variance_weights_long_only
        cov = sample_covariance(rendimientos_sinteticos)
        w_largo = min_variance_weights_long_only(cov)
        w_igual = equal_weights(cov.shape[0])
        assert (w_largo @ cov @ w_largo) <= (w_igual @ cov @ w_igual)

    def test_concentra_en_el_activo_tranquilo(self):
        """Sin correlacion, debe cargar el de menor varianza."""
        from src.risk import min_variance_weights_long_only
        w = min_variance_weights_long_only(np.diag([0.01, 0.04, 0.09]))
        assert w[0] > w[1] > w[2]

    def test_proyeccion_al_simplex(self):
        from src.risk import _project_to_simplex
        for v in [np.array([0.5, 0.3, 0.2]), np.array([2.0, -1.0, 0.5]),
                  np.array([-3.0, -1.0, -2.0])]:
            p = _project_to_simplex(v)
            assert np.isclose(p.sum(), 1.0)
            assert (p >= -1e-12).all()
