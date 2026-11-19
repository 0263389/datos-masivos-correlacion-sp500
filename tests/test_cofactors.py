"""Pruebas del modelo con cofactor.

Las pruebas centrales construyen series donde los coeficientes se conocen de
antemano -- se fija una beta en calma y otra en tension, se genera la serie, y
se verifica que el estimador recupera ambas. Si no recupera lo que se escondio
a proposito, no hay razon para creerle sobre datos reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cofactors import (
    CRITICO_5PCT,
    _ols_robusto,
    beta_con_interaccion,
    beta_simple,
    escalones_de_beta,
    mercado,
    niveles_de_tension,
    resumen_de_cofactores,
    tabla_de_cofactores,
    tension_por_volatilidad,
)

N = 3000


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture
def escenario(rng):
    """Serie con beta 0.8 en calma y 1.6 en tension. gamma verdadera = 0.8."""
    fechas = pd.bdate_range("2015-01-01", periods=N)
    r_mercado = pd.Series(rng.standard_normal(N) * 0.01, index=fechas)
    tension = pd.Series(np.arange(N) % 5 == 0, index=fechas)  # 20% tensos

    beta = np.where(tension, 1.6, 0.8)
    r_activo = pd.Series(
        beta * r_mercado.to_numpy() + 0.008 * rng.standard_normal(N), index=fechas
    )
    return r_activo, r_mercado, tension


class TestOlsRobusto:
    def test_recupera_coeficientes_conocidos(self, rng):
        X = np.column_stack([np.ones(N), rng.standard_normal(N), rng.standard_normal(N)])
        verdaderos = np.array([0.5, -1.5, 2.0])
        y = X @ verdaderos + 0.1 * rng.standard_normal(N)

        assert _ols_robusto(X, y)["coef"] == pytest.approx(verdaderos, abs=0.02)

    def test_con_varianza_constante_los_dos_errores_coinciden(self, rng):
        """Si el supuesto clasico se cumple, robusto y clasico deben parecerse."""
        X = np.column_stack([np.ones(N), rng.standard_normal(N)])
        y = X @ np.array([0.0, 1.0]) + 0.1 * rng.standard_normal(N)

        a = _ols_robusto(X, y)
        assert a["ee_robusto"][1] == pytest.approx(a["ee_clasico"][1], rel=0.15)

    def test_con_varianza_cambiante_el_robusto_es_mayor(self, rng):
        """La razon de ser del estimador robusto.

        El error se construye con desviacion proporcional al regresor, que es
        heterocedasticidad severa. El error estandar clasico la ignora y sale
        demasiado chico, lo que inflaria el estadistico t.
        """
        x = rng.standard_normal(N)
        X = np.column_stack([np.ones(N), x])
        y = x + np.abs(x) * rng.standard_normal(N)

        a = _ols_robusto(X, y)
        assert a["ee_robusto"][1] > 1.3 * a["ee_clasico"][1]

    def test_r2_entre_cero_y_uno(self, rng):
        X = np.column_stack([np.ones(N), rng.standard_normal(N)])
        y = rng.standard_normal(N)
        assert 0 <= _ols_robusto(X, y)["r2"] <= 1

    def test_r2_uno_sin_error(self, rng):
        x = rng.standard_normal(N)
        X = np.column_stack([np.ones(N), x])
        assert _ols_robusto(X, 3 * x + 1)["r2"] == pytest.approx(1.0)

    def test_rechaza_menos_observaciones_que_parametros(self):
        with pytest.raises(ValueError, match="mas observaciones"):
            _ols_robusto(np.ones((2, 3)), np.ones(2))


class TestBetaConInteraccion:
    def test_recupera_las_dos_betas(self, escenario):
        """La prueba principal: se escondieron 0.8 y 1.6."""
        r = beta_con_interaccion(*escenario)
        assert r["beta_calma"] == pytest.approx(0.8, abs=0.05)
        assert r["beta_tension"] == pytest.approx(1.6, abs=0.08)
        assert r["gamma"] == pytest.approx(0.8, abs=0.10)

    def test_gamma_es_significativa_cuando_el_escalon_existe(self, escenario):
        assert beta_con_interaccion(*escenario)["significativo"]
        assert abs(beta_con_interaccion(*escenario)["t_gamma"]) > CRITICO_5PCT

    def test_gamma_no_es_significativa_sin_escalon(self, rng):
        """El otro lado. Sin esta prueba, un estimador que siempre declarara
        significancia tambien pasaria la anterior.
        """
        fechas = pd.bdate_range("2015-01-01", periods=N)
        m = pd.Series(rng.standard_normal(N) * 0.01, index=fechas)
        tension = pd.Series(np.arange(N) % 5 == 0, index=fechas)
        y = pd.Series(1.1 * m.to_numpy() + 0.008 * rng.standard_normal(N), index=fechas)

        r = beta_con_interaccion(y, m, tension)
        assert not r["significativo"]
        assert abs(r["gamma"]) < 0.15

    def test_el_cofactor_explica_mas_que_la_beta_unica(self, escenario):
        """Agregar el termino de interaccion nunca puede empeorar el ajuste."""
        con = beta_con_interaccion(*escenario)["r2"]
        sin = beta_simple(escenario[0], escenario[1])["r2"]
        assert con >= sin

    def test_la_beta_unica_queda_entre_las_dos(self, escenario):
        """Es el promedio ponderado por dias de cada regimen, asi que no puede
        caer fuera del intervalo -- y por eso describe mal a los dos.
        """
        r = beta_con_interaccion(*escenario)
        unica = beta_simple(escenario[0], escenario[1])["beta"]
        assert r["beta_calma"] < unica < r["beta_tension"]

    def test_rechaza_tension_de_un_solo_regimen(self, escenario):
        r_activo, r_mercado, _ = escenario
        constante = pd.Series(False, index=r_mercado.index)
        with pytest.raises(ValueError, match="los dos regimenes"):
            beta_con_interaccion(r_activo, r_mercado, constante)

    def test_rechaza_serie_corta(self, escenario):
        a, m, d = (s.head(50) for s in escenario)
        with pytest.raises(ValueError, match="al menos 100"):
            beta_con_interaccion(a, m, d)


class TestTension:
    def test_marca_la_fraccion_pedida(self, rng):
        m = pd.Series(rng.standard_normal(N) * 0.01,
                      index=pd.bdate_range("2015-01-01", periods=N))
        t = tension_por_volatilidad(m, cuantil=0.80)
        assert t.mean() == pytest.approx(0.20, abs=0.02)

    def test_los_dias_tensos_son_mas_volatiles(self, rng):
        """Verificacion de sentido: si no lo fueran, la clasificacion no sirve."""
        fechas = pd.bdate_range("2015-01-01", periods=N)
        escala = np.where(np.arange(N) % 500 < 100, 0.03, 0.008)
        m = pd.Series(escala * rng.standard_normal(N), index=fechas)

        t = tension_por_volatilidad(m)
        assert m[t].std() > 2 * m[~t].std()

    def test_la_variante_expandiendo_no_usa_el_futuro(self, rng):
        """Recalcular con datos truncados no debe cambiar el pasado.

        Es la prueba que distingue las dos variantes: con el cuantil de muestra
        completa, agregar dias al final puede reclasificar dias anteriores.
        """
        fechas = pd.bdate_range("2015-01-01", periods=N)
        m = pd.Series(rng.standard_normal(N) * 0.01, index=fechas)
        params = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90}

        completo = tension_por_volatilidad(m, parametros=params, expandiendo=True)
        truncado = tension_por_volatilidad(
            m.head(2000), parametros=params, expandiendo=True
        )
        pd.testing.assert_series_equal(completo.head(2000), truncado)

    def test_niveles_reparte_en_grupos_iguales(self, rng):
        m = pd.Series(rng.standard_normal(N) * 0.01,
                      index=pd.bdate_range("2015-01-01", periods=N))
        niveles = niveles_de_tension(m, n_grupos=5)
        assert sorted(niveles.unique()) == [1, 2, 3, 4, 5]
        assert niveles.value_counts().std() < 2


class TestEscalones:
    def test_detecta_betas_crecientes_por_nivel(self, rng):
        """Serie donde la beta sube con el nivel de volatilidad del mercado."""
        fechas = pd.bdate_range("2015-01-01", periods=N)
        bloque = np.arange(N) // (N // 5)
        escala = np.array([0.005, 0.008, 0.012, 0.018, 0.030])[np.clip(bloque, 0, 4)]
        m = pd.Series(escala * rng.standard_normal(N), index=fechas)

        betas_reales = np.array([0.6, 0.8, 1.0, 1.2, 1.4])[np.clip(bloque, 0, 4)]
        y = pd.Series(betas_reales * m.to_numpy() + 0.004 * rng.standard_normal(N),
                      index=fechas)

        tabla = escalones_de_beta(y, m, niveles_de_tension(m, 5))
        assert len(tabla) == 5
        assert tabla["beta"].is_monotonic_increasing
        assert tabla["vol_mercado"].is_monotonic_increasing

    def test_beta_plana_cuando_no_cambia(self, rng):
        fechas = pd.bdate_range("2015-01-01", periods=N)
        m = pd.Series(rng.standard_normal(N) * 0.01, index=fechas)
        y = pd.Series(0.9 * m.to_numpy() + 0.005 * rng.standard_normal(N), index=fechas)

        tabla = escalones_de_beta(y, m, niveles_de_tension(m, 5))
        assert tabla["beta"].std() < 0.15


class TestTablaCompleta:
    @pytest.fixture
    def panel(self, rng):
        fechas = pd.bdate_range("2015-01-01", periods=N)
        factor = rng.standard_normal(N) * 0.01
        datos = {
            f"T{i}": (0.5 + 0.3 * i) * factor + 0.008 * rng.standard_normal(N)
            for i in range(6)
        }
        return pd.DataFrame(datos, index=fechas)

    def test_una_fila_por_activo_ordenada_por_gamma(self, panel):
        tabla = tabla_de_cofactores(panel)
        assert len(tabla) == len(panel.columns)
        assert tabla["gamma"].tolist() == sorted(tabla["gamma"].tolist(), reverse=True)

    def test_las_betas_promedian_uno(self, panel):
        """Identidad, no hallazgo empirico.

        El mercado se define como el promedio equiponderado de estos mismos
        activos, asi que la beta promedio contra el es exactamente 1 en
        cualquier submuestra, y por lo tanto la gamma promedio es cero.

        La consecuencia practica importa: el agregado **no puede** revelar nada
        sobre el efecto del regimen. Toda la informacion esta en la dispersion,
        y un resumen que solo reportara promedios concluiria que no pasa nada.
        """
        tabla = tabla_de_cofactores(panel)
        assert tabla["beta_calma"].mean() == pytest.approx(1.0, abs=1e-9)
        assert tabla["beta_tension"].mean() == pytest.approx(1.0, abs=1e-9)
        assert tabla["gamma"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_beta_unica_entre_las_dos_para_cada_activo(self, panel):
        tabla = tabla_de_cofactores(panel)
        entre = (
            (tabla["beta_unica"] >= tabla[["beta_calma", "beta_tension"]].min(axis=1))
            & (tabla["beta_unica"] <= tabla[["beta_calma", "beta_tension"]].max(axis=1))
        )
        assert entre.all()

    def test_mercado_es_el_promedio(self, panel):
        pd.testing.assert_series_equal(mercado(panel), panel.mean(axis=1))

    def test_resumen_tiene_las_medidas_clave(self, panel):
        resumen = resumen_de_cofactores(tabla_de_cofactores(panel))
        medidas = " ".join(resumen["medida"])
        assert "significativa" in medidas
        assert "robusto" in medidas

    def test_el_resumen_queda_definido_cuando_hay_significativas(self, rng):
        """Con un escalon real, ninguna celda del resumen debe quedar sin valor."""
        fechas = pd.bdate_range("2015-01-01", periods=N)
        factor = pd.Series(rng.standard_normal(N) * 0.01, index=fechas)
        tension = tension_por_volatilidad(factor)

        datos = {}
        for i in range(6):
            beta = np.where(tension, 0.5 + 0.4 * i, 1.5 - 0.2 * i)
            datos[f"T{i}"] = beta * factor.to_numpy() + 0.006 * rng.standard_normal(N)
        panel = pd.DataFrame(datos, index=fechas)

        tabla = tabla_de_cofactores(panel, factor, tension)
        assert tabla["significativo"].any(), "el escenario deberia dar significancia"
        assert resumen_de_cofactores(tabla)["valor"].notna().all()

    def test_la_fraccion_positiva_es_nula_sin_significativas(self, panel):
        """Cero de cero es indefinido, no cero. El resumen debe decirlo con NaN
        en lugar de inventar un 0% que se leeria como un hallazgo.
        """
        tabla = tabla_de_cofactores(panel)
        tabla["significativo"] = False
        resumen = resumen_de_cofactores(tabla).set_index("medida")
        assert np.isnan(resumen.loc["de esas, con gamma positiva", "valor"])
