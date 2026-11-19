"""Pruebas de las medidas de dependencia no lineal.

La estrategia es la misma que en test_volatility.py: construir series donde la
respuesta correcta se conoce de antemano, y verificar que el estimador la
encuentra. Incluye los dos lados de cada afirmacion -- que detecta lo que hay y
que no inventa lo que no hay -- porque una prueba de un solo lado la pasaria un
estimador que siempre responde lo mismo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nonlinear import (
    BINS,
    asimetria_de_colas,
    correlacion_equivalente,
    dependencia_de_cola,
    informacion_mutua,
    piso_de_brecha,
    resumen_no_lineal,
    sesgo_por_permutacion,
    tabla_no_lineal,
)

N = 3000


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def panel(rng):
    """Panel con estructura conocida: tres activos correlacionados y uno libre."""
    factor = rng.standard_normal(N)
    datos = {
        "A": 0.8 * factor + 0.6 * rng.standard_normal(N),
        "B": 0.8 * factor + 0.6 * rng.standard_normal(N),
        "C": 0.3 * factor + 0.95 * rng.standard_normal(N),
        "D": rng.standard_normal(N),
    }
    fechas = pd.bdate_range("2015-01-01", periods=N)
    return pd.DataFrame(datos, index=fechas) * 0.01


class TestInformacionMutua:
    def test_independientes_dan_casi_cero(self, rng):
        """Series sin relacion: la MI corregida debe quedar cerca de cero."""
        mi = informacion_mutua(rng.standard_normal(N), rng.standard_normal(N))
        assert 0 <= mi < 0.01

    def test_detecta_relacion_en_U_que_pearson_no_ve(self, rng):
        """El caso que justifica todo el modulo.

        Con y = x^2 la relacion es perfecta y deterministica, pero la
        correlacion de Pearson vale casi cero porque la nube es simetrica: la
        mejor recta es horizontal.
        """
        x = rng.standard_normal(N)
        y = x**2 + 0.3 * rng.standard_normal(N)

        pearson = abs(np.corrcoef(x, y)[0, 1])
        equivalente = correlacion_equivalente(informacion_mutua(x, y))

        assert pearson < 0.10, "la U deberia ser invisible para Pearson"
        assert equivalente > 0.60, "la informacion mutua deberia verla"
        assert equivalente > 5 * pearson

    def test_es_simetrica(self, rng):
        x, y = rng.standard_normal(N), rng.standard_normal(N)
        assert informacion_mutua(x, y) == pytest.approx(informacion_mutua(y, x))

    def test_crece_con_la_dependencia(self, rng):
        """Mas senial compartida debe dar mas informacion mutua."""
        f = rng.standard_normal(N)
        debil = 0.2 * f + 0.98 * rng.standard_normal(N)
        fuerte = 0.9 * f + 0.44 * rng.standard_normal(N)
        assert informacion_mutua(f, fuerte) > informacion_mutua(f, debil)

    def test_una_serie_contra_si_misma_es_maxima(self, rng):
        """El tope lo impone la discretizacion: log(bins) nats."""
        x = rng.standard_normal(N)
        mi = informacion_mutua(x, x, corregir_sesgo=False)
        assert mi == pytest.approx(np.log(BINS), rel=0.02)

    def test_rechaza_serie_corta(self, rng):
        with pytest.raises(ValueError, match="al menos 100"):
            informacion_mutua(rng.standard_normal(50), rng.standard_normal(50))


class TestCorreccionDeSesgo:
    def test_el_sesgo_analitico_coincide_con_el_empirico(self, rng):
        """La correccion (bins-1)^2/(2n) debe igualar lo que mide permutar.

        Es la prueba que vuelve defendible restar una formula en lugar de
        permutar 300 veces por par, que seria imposible en 104,653 pares.
        """
        x = rng.standard_normal(N)
        y = 0.5 * x + 0.87 * rng.standard_normal(N)
        s = sesgo_por_permutacion(x, y, repeticiones=300)

        assert s["sesgo_analitico"] == pytest.approx(
            s["sesgo_empirico"], abs=3 * s["desviacion_empirica"]
        )

    def test_la_dependencia_real_sobresale_del_sesgo(self, rng):
        x = rng.standard_normal(N)
        y = 0.6 * x + 0.8 * rng.standard_normal(N)
        assert sesgo_por_permutacion(x, y, repeticiones=200)["z"] > 10

    def test_sin_dependencia_no_sobresale(self, rng):
        """El otro lado: ruido independiente no debe dar un z alto."""
        s = sesgo_por_permutacion(
            rng.standard_normal(N), rng.standard_normal(N), repeticiones=200
        )
        assert abs(s["z"]) < 4


class TestCorrelacionEquivalente:
    def test_cero_en_cero(self):
        assert correlacion_equivalente(0.0) == 0.0

    def test_invierte_la_formula_gaussiana(self):
        """Partiendo de la MI teorica de una normal bivariada debe devolver rho."""
        for rho in (0.3, 0.5, 0.7, 0.9):
            mi_teorica = -0.5 * np.log(1 - rho**2)
            assert correlacion_equivalente(mi_teorica) == pytest.approx(rho, abs=1e-9)

    def test_es_creciente(self):
        valores = [correlacion_equivalente(m) for m in (0.05, 0.2, 0.5, 1.0)]
        assert valores == sorted(valores)

    def test_nunca_pasa_de_uno(self):
        assert correlacion_equivalente(50.0) <= 1.0


class TestDependenciaDeCola:
    def test_independencia_da_el_cuantil(self, rng):
        """Sin relacion, P(Y extremo | X extremo) debe valer el cuantil.

        La tolerancia no es arbitraria: la probabilidad se estima sobre los
        ``N * cuantil`` dias extremos, asi que su error estandar es
        ``sqrt(q(1-q)/(N*q))``. Se admiten tres de esos. Con una tolerancia mas
        apretada la prueba fallaria por azar segun la semilla, no por un error.
        """
        q, dias = 0.10, N * 0.10
        tolerancia = 3 * np.sqrt(q * (1 - q) / dias)

        d = dependencia_de_cola(rng.standard_normal(N), rng.standard_normal(N), q)
        assert d["probabilidad"] == pytest.approx(q, abs=tolerancia)
        assert d["exceso"] == pytest.approx(1.0, abs=tolerancia / q)

    def test_series_identicas_dan_uno(self, rng):
        x = rng.standard_normal(N)
        assert dependencia_de_cola(x, x, 0.10)["probabilidad"] == pytest.approx(1.0)

    def test_detecta_dependencia_solo_en_la_cola(self, rng):
        """Construccion donde las caidas van juntas y las subidas no.

        Es el caso que la correlacion no puede expresar, porque es un solo
        coeficiente para toda la nube.

        En los dias en que ``x`` cae a su peor decil, ``y`` se desploma a un
        nivel fijo; el resto del tiempo son independientes. El desplome de
        ``y`` es un nivel fijo y **no** una copia del valor de ``x``, y eso es
        deliberado: copiar ``x`` metería en ``y`` los mismos valores extremos y
        correria su propio umbral del decil hacia abajo, con lo que parte de los
        dias de crisis dejaria de contar como extremo de ``y``. El fenomeno a
        probar es la coincidencia de eventos extremos, no el valor que toman.
        """
        x = rng.standard_normal(N)
        y = rng.standard_normal(N)
        crisis = x < np.quantile(x, 0.10)
        y = np.where(crisis, -3 + 0.1 * rng.standard_normal(N), y)

        a = asimetria_de_colas(x, y, 0.10)
        assert a["prob_caidas"] > 0.90
        assert a["prob_subidas"] < 0.25
        assert a["asimetria"] > 0.5
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.60, (
            "la correlacion promedia los dos regimenes y no revela la asimetria"
        )

    def test_simetrica_da_asimetria_nula(self, rng):
        """El otro lado: una relacion lineal simetrica no debe dar asimetria."""
        x = rng.standard_normal(N)
        y = 0.7 * x + 0.71 * rng.standard_normal(N)
        assert abs(asimetria_de_colas(x, y, 0.10)["asimetria"]) < 0.10

    def test_rechaza_cola_invalida(self, rng):
        with pytest.raises(ValueError, match="inferior"):
            dependencia_de_cola(rng.standard_normal(N), rng.standard_normal(N),
                                cola="medio")


class TestTablaYPiso:
    def test_una_fila_por_par_sin_repetir(self, panel):
        tabla = tabla_no_lineal(panel)
        assert len(tabla) == 4 * 3 // 2
        assert not tabla.duplicated(["activo_a", "activo_b"]).any()

    def test_ordenada_por_brecha(self, panel):
        brechas = tabla_no_lineal(panel)["brecha"].tolist()
        assert brechas == sorted(brechas, reverse=True)

    def test_pearson_coincide_con_numpy(self, panel):
        tabla = tabla_no_lineal(panel).set_index(["activo_a", "activo_b"])
        esperado = panel.corr()
        for (a, b), fila in tabla.iterrows():
            assert fila["pearson"] == pytest.approx(esperado.loc[a, b], abs=1e-10)

    def test_el_par_mas_dependiente_tiene_mas_info_mutua(self, panel):
        """A y B comparten el factor con peso 0.8; D no comparte nada."""
        tabla = tabla_no_lineal(panel).set_index(["activo_a", "activo_b"])
        assert tabla.loc[("A", "B"), "info_mutua"] > tabla.loc[("A", "D"), "info_mutua"]

    def test_el_piso_es_pequenio_pero_no_cero(self, panel):
        """El piso calibra la brecha: sin el, un 0.03 no se puede interpretar."""
        piso = piso_de_brecha(panel, repeticiones=100)
        assert 0 < piso["percentil_95"] < 0.25
        assert piso["percentil_95"] > piso["media"]

    def test_un_panel_lineal_no_supera_su_propio_piso(self, panel):
        """Prueba clave de calibracion.

        El panel se construyo con relaciones estrictamente lineales. La
        proporcion de pares que supera el percentil 95 del piso no deberia
        alejarse del 5% que corresponde por construccion.
        """
        piso = piso_de_brecha(panel, repeticiones=300)["percentil_95"]
        supera = (tabla_no_lineal(panel)["brecha"] > piso).mean()
        assert supera <= 0.35, f"demasiados falsos positivos: {supera:.0%}"

    def test_resumen_con_piso_agrega_el_umbral(self, panel):
        tabla = tabla_no_lineal(panel)
        sin = resumen_no_lineal(tabla)
        con = resumen_no_lineal(tabla, piso=0.05)
        assert len(con) == len(sin) + 3
        assert "umbral" in con.iloc[0]["medida"]

    def test_rechaza_panel_corto(self, panel):
        with pytest.raises(ValueError, match="al menos 100"):
            tabla_no_lineal(panel.head(50))
