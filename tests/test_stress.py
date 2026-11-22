"""Pruebas de las medidas de estres.

Igual que en el resto del proyecto, cada afirmacion se prueba por sus dos
lados: que la medida detecta lo que debe cuando esta, y que no lo inventa
cuando no esta. Varias construcciones fijan ademas identidades algebraicas
-el error del choque es exactamente gamma por el choque, los aportes suman el
agregado- porque una identidad que se rompe delata un error de signo o de
ponderacion que ninguna prueba estadistica atraparia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stress import (
    ESCENARIOS,
    choque_de_mercado,
    detalle_del_choque,
    escenario_historico,
    maxima_caida,
    peores_ventanas,
    riesgo_por_regimen,
    serie_portafolio,
    tabla_de_escenarios,
    var_historico,
)

N = 2000


@pytest.fixture
def rng():
    return np.random.default_rng(13)


@pytest.fixture
def panel(rng):
    fechas = pd.bdate_range("2016-01-01", periods=N)
    factor = rng.standard_normal(N) * 0.01
    datos = {
        f"T{i}": (0.6 + 0.25 * i) * factor + 0.008 * rng.standard_normal(N)
        for i in range(4)
    }
    return pd.DataFrame(datos, index=fechas)


@pytest.fixture
def pesos():
    return np.array([0.4, 0.3, 0.2, 0.1])


@pytest.fixture
def cofactores():
    """Tabla con gamma conocida, para fijar las identidades del choque."""
    return pd.DataFrame({
        "ticker": ["T0", "T1", "T2", "T3"],
        "sector": ["A", "A", "B", "B"],
        "beta_unica": [0.60, 0.85, 1.10, 1.35],
        "beta_calma": [0.55, 0.83, 1.12, 1.40],
        "beta_tension": [0.80, 0.95, 1.05, 1.20],
        "gamma": [0.25, 0.12, -0.07, -0.20],
    })


class TestSeriePortafolio:
    def test_un_solo_activo_devuelve_ese_activo(self, panel):
        s = serie_portafolio(panel, np.array([1.0]), ["T0"])
        pd.testing.assert_series_equal(s, panel["T0"].rename("portafolio"),
                                       check_exact=False, atol=1e-12)

    def test_conserva_el_indice(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        pd.testing.assert_index_equal(s.index, panel.index)

    def test_pondera_sobre_rendimientos_simples(self, panel, pesos):
        """La suma ponderada se hace en simples, no en logaritmicos.

        El logaritmo no es lineal, asi que ambas definiciones difieren. La
        prueba fija cual se usa, y de paso verifica que la diferencia existe:
        si fueran identicas, la distincion del docstring seria decorativa.
        """
        tickers = list(panel.columns)
        s = serie_portafolio(panel, pesos, tickers)
        correcto = np.log1p(np.expm1(panel[tickers].to_numpy()) @ pesos)
        ingenuo = panel[tickers].to_numpy() @ pesos

        assert np.allclose(s.to_numpy(), correcto, atol=1e-12)
        assert not np.allclose(s.to_numpy(), ingenuo, atol=1e-10)

    def test_rechaza_pesos_que_no_suman_uno(self, panel):
        with pytest.raises(ValueError, match="deben sumar 1"):
            serie_portafolio(panel, np.array([0.5, 0.3, 0.1, 0.05]), list(panel.columns))

    def test_rechaza_desajuste_de_longitud(self, panel):
        with pytest.raises(ValueError, match="pesos para"):
            serie_portafolio(panel, np.array([1.0]), list(panel.columns))


class TestVaR:
    def test_el_expected_shortfall_nunca_es_menor_que_el_var(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        for nivel in (0.90, 0.95, 0.99):
            v = var_historico(s, nivel)
            assert v["expected_shortfall"] >= v["var"]

    def test_crece_con_el_nivel(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        assert var_historico(s, 0.99)["var"] > var_historico(s, 0.95)["var"]

    def test_la_cola_tiene_el_tamanio_esperado(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        v = var_historico(s, 0.95)
        assert v["dias_en_la_cola"] == pytest.approx(0.05 * len(s), rel=0.12)

    def test_colas_pesadas_elevan_la_razon_es_var(self, rng):
        """Distingue una distribucion normal de una con colas pesadas.

        Es la propiedad que justifica reportar el Expected Shortfall: el VaR
        solo mira el borde de la cola y apenas se mueve, mientras que el ES
        promedia lo que hay detras y si lo detecta.
        """
        fechas = pd.bdate_range("2016-01-01", periods=5000)
        normal = pd.Series(rng.standard_normal(5000) * 0.01, index=fechas)
        pesada = pd.Series(rng.standard_t(2.5, 5000) * 0.006, index=fechas)

        assert (var_historico(pesada)["razon_es_var"]
                > var_historico(normal)["razon_es_var"])

    def test_rechaza_nivel_invalido(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        with pytest.raises(ValueError, match="entre 0.5 y 1"):
            var_historico(s, 1.5)

    def test_rechaza_serie_corta(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        with pytest.raises(ValueError, match="al menos 100"):
            var_historico(s.head(50))


class TestMaximaCaida:
    def test_recupera_una_caida_construida(self):
        """Serie que sube 10%, cae 20% y se recupera. La caida es -20%."""
        fechas = pd.bdate_range("2020-01-01", periods=31)
        tramos = np.concatenate([
            np.full(10, np.log(1.10) / 10),    # sube
            np.full(10, np.log(0.80) / 10),    # cae
            np.full(11, np.log(1.30) / 11),    # recupera y supera
        ])
        md = maxima_caida(pd.Series(tramos, index=fechas))

        assert md["caida"] == pytest.approx(-0.20, abs=1e-9)
        assert md["dias_de_caida"] == 10
        assert md["recuperacion"] is not None

    def test_serie_siempre_al_alza_no_tiene_caida(self):
        fechas = pd.bdate_range("2020-01-01", periods=50)
        md = maxima_caida(pd.Series(np.full(50, 0.001), index=fechas))
        assert md["caida"] == pytest.approx(0.0, abs=1e-12)

    def test_sin_recuperacion_lo_reporta_como_ausente(self):
        fechas = pd.bdate_range("2020-01-01", periods=30)
        serie = pd.Series(np.concatenate([np.full(10, 0.01), np.full(20, -0.01)]),
                          index=fechas)
        assert maxima_caida(serie)["recuperacion"] is None

    def test_rechaza_serie_vacia(self):
        with pytest.raises(ValueError, match="vacia"):
            maxima_caida(pd.Series(dtype=float))


class TestEscenarios:
    def test_recorta_la_ventana_pedida(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        e = escenario_historico(s, "2017-01-01", "2017-03-31", "prueba")
        assert e["inicio"] >= pd.Timestamp("2017-01-01")
        assert e["fin"] <= pd.Timestamp("2017-03-31")
        assert e["escenario"] == "prueba"

    def test_la_caida_maxima_no_supera_al_rendimiento_del_tramo(self, panel, pesos):
        """La caida pico a valle siempre es al menos tan mala como el total."""
        s = serie_portafolio(panel, pesos, list(panel.columns))
        e = escenario_historico(s, "2017-01-01", "2017-12-31")
        assert e["caida_maxima"] <= min(e["rendimiento"], 0) + 1e-12

    def test_ventana_fuera_del_periodo(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        with pytest.raises(ValueError, match="No hay datos"):
            escenario_historico(s, "1990-01-01", "1990-12-31")

    def test_la_tabla_omite_los_escenarios_sin_datos(self, panel, pesos):
        """El panel sintetico no cubre todas las fechas de ESCENARIOS."""
        s = serie_portafolio(panel, pesos, list(panel.columns))
        tabla = tabla_de_escenarios(s)
        assert len(tabla) <= len(ESCENARIOS)
        if len(tabla) > 1:
            assert tabla["rendimiento"].is_monotonic_increasing

    def test_las_peores_ventanas_no_se_traslapan(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        t = peores_ventanas(s, dias=21, n=4, separacion=90)
        fechas = t["fin"].tolist()
        assert all(abs((a - b).days) >= 90
                   for i, a in enumerate(fechas) for b in fechas[i + 1:])

    def test_la_peor_ventana_es_la_peor(self, panel, pesos):
        s = serie_portafolio(panel, pesos, list(panel.columns))
        t = peores_ventanas(s, dias=21, n=3)
        assert t["rendimiento"].is_monotonic_increasing
        assert t["rendimiento"].iloc[0] == pytest.approx(
            float(np.expm1(s.rolling(21).sum().min())), rel=1e-9)


class TestRiesgoPorRegimen:
    def test_la_tension_es_mas_volatil_que_la_calma(self, rng):
        """Verificacion de sentido sobre datos construidos con dos regimenes."""
        fechas = pd.bdate_range("2016-01-01", periods=N)
        tension = pd.Series(np.arange(N) % 5 == 0, index=fechas)
        escala = np.where(tension, 0.030, 0.007)

        factor = escala * rng.standard_normal(N)
        panel = pd.DataFrame(
            {f"T{i}": (0.8 + 0.2 * i) * factor + 0.004 * rng.standard_normal(N)
             for i in range(3)}, index=fechas)

        t = riesgo_por_regimen(panel, np.array([1 / 3] * 3), list(panel.columns), tension)
        vol = t.set_index("regimen")["vol_portafolio"]
        assert vol["Tensión"] > 2 * vol["Calma"]

    def test_las_sesiones_de_los_regimenes_suman_el_total(self, panel, pesos, rng):
        tension = pd.Series(rng.random(len(panel)) < 0.2, index=panel.index)
        t = riesgo_por_regimen(panel, pesos, list(panel.columns), tension).set_index("regimen")
        assert t.loc["Calma", "sesiones"] + t.loc["Tensión", "sesiones"] == len(panel)

    def test_la_comparacion_contra_el_periodo_completo_es_cero_en_si_misma(
            self, panel, pesos, rng):
        tension = pd.Series(rng.random(len(panel)) < 0.2, index=panel.index)
        t = riesgo_por_regimen(panel, pesos, list(panel.columns), tension).set_index("regimen")
        assert t.loc["Periodo completo", "vs_completo"] == pytest.approx(0.0)


class TestChoqueDeMercado:
    def test_el_error_es_el_salto_contra_la_beta_unica(self, pesos, cofactores):
        """Identidad algebraica, no aproximacion.

        error = Σ wᵢ·β_tensiónᵢ·c − Σ wᵢ·β_únicaᵢ·c = Σ wᵢ(β_tensiónᵢ − β_únicaᵢ)·c
        """
        choque = -0.10
        tk = ["T0", "T1", "T2", "T3"]
        c = choque_de_mercado(pesos, tk, cofactores, choque)
        salto = cofactores["beta_tension"].to_numpy() - cofactores["beta_unica"].to_numpy()

        assert c["error_del_modelo_estandar"] == pytest.approx(
            float(pesos @ salto) * choque, abs=1e-14)
        assert c["salto_vs_beta_unica"] == pytest.approx(float(pesos @ salto), abs=1e-14)

    def test_gamma_y_el_error_son_cantidades_distintas(self, pesos, cofactores):
        """La distincion que el modulo tiene que mantener separada.

        gamma es el salto de calma a tension. La beta unica se estima sobre
        todos los dias, de modo que ya absorbe parte de ese salto y el error
        real de usarla es menor. Confundirlas exageraria el problema.
        """
        tk = ["T0", "T1", "T2", "T3"]
        c = choque_de_mercado(pesos, tk, cofactores, -0.10)

        assert c["gamma_cartera"] == pytest.approx(
            float(pesos @ cofactores["gamma"].to_numpy()), abs=1e-14)
        assert c["beta_calma"] < c["beta_unica"] < c["beta_tension"], (
            "la beta unica deberia caer entre las dos del modelo con cofactor")
        assert abs(c["salto_vs_beta_unica"]) < abs(c["gamma_cartera"])

    def test_sin_cambio_de_regimen_los_dos_modelos_coinciden(self, pesos, cofactores):
        """El otro lado: si gamma es cero, no hay error que reportar."""
        plano = cofactores.copy()
        plano["gamma"] = 0.0
        plano["beta_calma"] = plano["beta_unica"]
        plano["beta_tension"] = plano["beta_unica"]

        c = choque_de_mercado(pesos, ["T0", "T1", "T2", "T3"], plano, -0.10)
        assert c["error_del_modelo_estandar"] == pytest.approx(0.0, abs=1e-14)
        assert c["subestimacion_relativa"] == pytest.approx(0.0, abs=1e-12)

    def test_la_subestimacion_relativa_no_depende_del_tamanio_del_choque(
            self, pesos, cofactores):
        """Ambas proyecciones son lineales en el choque, asi que su razon es fija.

        Si esta prueba fallara, alguna de las dos dejo de ser lineal y la
        comparacion entre modelos ya no seria limpia.
        """
        tk = ["T0", "T1", "T2", "T3"]
        razones = [choque_de_mercado(pesos, tk, cofactores, c)["subestimacion_relativa"]
                   for c in (-0.05, -0.10, -0.30)]
        assert np.allclose(razones, razones[0], atol=1e-12)

    def test_una_caida_produce_perdida(self, pesos, cofactores):
        c = choque_de_mercado(pesos, ["T0", "T1", "T2", "T3"], cofactores, -0.10)
        assert c["perdida_modelo_estandar"] < 0
        assert c["perdida_modelo_condicional"] < 0

    def test_rechaza_activos_sin_beta_estimada(self, pesos, cofactores):
        with pytest.raises(ValueError, match="Sin beta condicional"):
            choque_de_mercado(pesos, ["T0", "T1", "T2", "DESCONOCIDO"],
                              cofactores, -0.10)

    def test_el_desglose_suma_el_agregado(self, pesos, cofactores):
        """Cada posicion aporta su parte y las partes reconstruyen el total."""
        tk = ["T0", "T1", "T2", "T3"]
        choque = -0.10
        c = choque_de_mercado(pesos, tk, cofactores, choque)
        d = detalle_del_choque(pesos, tk, cofactores, choque)

        assert d["aporte_estandar"].sum() == pytest.approx(
            c["perdida_modelo_estandar"], abs=1e-14)
        assert d["aporte_condicional"].sum() == pytest.approx(
            c["perdida_modelo_condicional"], abs=1e-14)
        assert d["error"].sum() == pytest.approx(
            c["error_del_modelo_estandar"], abs=1e-14)

    def test_el_desglose_ordena_por_error(self, pesos, cofactores):
        d = detalle_del_choque(pesos, ["T0", "T1", "T2", "T3"], cofactores, -0.10)
        assert d["error"].is_monotonic_increasing
