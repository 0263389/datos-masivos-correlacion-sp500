"""Pruebas de la correlacion movil contra el mercado.

Mismo criterio que ``test_risk.py``: sin Spark y sin internet, sobre datos
sinteticos cuya respuesta se conoce de antemano.

Correr con:

    ./.venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    correlation_dispersion,
    fisher_z,
    rolling_market_correlation,
)

VENTANA = 60  # ventana corta para que las pruebas corran rapido


@pytest.fixture
def panel_sintetico() -> tuple[pd.DataFrame, pd.Series]:
    """400 dias x 6 acciones con exposicion al mercado conocida.

    Dos columnas son casos limite con respuesta exacta: ``replica`` es el
    mercado mismo (correlacion 1) y ``espejo`` es su negativo (correlacion
    -1). Las otras cuatro son ``beta * mercado + ruido propio``, el modelo de
    un factor del Entregable 2, con exposiciones decrecientes.
    """
    rng = np.random.default_rng(42)
    fechas = pd.bdate_range("2020-01-01", periods=400, name="date")

    mercado = pd.Series(0.01 * rng.standard_normal(len(fechas)), index=fechas,
                        name="mkt_return")

    columnas = {"replica": mercado, "espejo": -mercado}
    for beta in (1.0, 0.6, 0.3, 0.0):
        ruido = 0.01 * rng.standard_normal(len(fechas))
        columnas[f"beta_{beta}"] = beta * mercado + ruido

    return pd.DataFrame(columnas, index=fechas), mercado


class TestCorrelacionMovilContraElMercado:
    def test_recupera_la_correlacion_perfecta(self, panel_sintetico):
        """Una accion que ES el mercado tiene que dar exactamente 1.0."""
        panel, mercado = panel_sintetico
        rc = rolling_market_correlation(panel, mercado, window=VENTANA)
        assert np.allclose(rc["replica"], 1.0)

    def test_recupera_la_correlacion_inversa(self, panel_sintetico):
        panel, mercado = panel_sintetico
        rc = rolling_market_correlation(panel, mercado, window=VENTANA)
        assert np.allclose(rc["espejo"], -1.0)

    def test_la_exposicion_ordena_las_correlaciones(self, panel_sintetico):
        """A mayor beta, mayor correlacion con el factor.

        Es la prueba de que la funcion mide lo que dice medir: no basta con
        que devuelva numeros en [-1, 1], tienen que ordenarse como el modelo.
        """
        panel, mercado = panel_sintetico
        medias = rolling_market_correlation(panel, mercado, window=VENTANA).mean()
        assert medias["beta_1.0"] > medias["beta_0.6"] > medias["beta_0.3"]
        assert abs(medias["beta_0.0"]) < 0.15  # independiente del mercado

    def test_coincide_con_el_calculo_columna_por_columna(self, panel_sintetico):
        """El atajo vectorizado debe dar lo mismo que el calculo directo."""
        panel, mercado = panel_sintetico
        rc = rolling_market_correlation(panel, mercado, window=VENTANA)
        for col in panel.columns:
            directo = panel[col].rolling(VENTANA).corr(mercado).dropna()
            assert np.allclose(rc[col].dropna(), directo)

    def test_descarta_solo_el_periodo_de_calentamiento(self, panel_sintetico):
        """Las primeras ``window - 1`` fechas no tienen ventana completa."""
        panel, mercado = panel_sintetico
        rc = rolling_market_correlation(panel, mercado, window=VENTANA)
        assert len(rc) == len(panel) - VENTANA + 1
        assert rc.index[0] == panel.index[VENTANA - 1]

    def test_acepta_el_factor_como_dataframe_de_una_columna(self, panel_sintetico):
        """El factor sale de Spark como tabla, y a veces llega con su columna."""
        panel, mercado = panel_sintetico
        como_tabla = mercado.to_frame()
        assert np.allclose(
            rolling_market_correlation(panel, como_tabla, window=VENTANA),
            rolling_market_correlation(panel, mercado, window=VENTANA),
        )

    def test_rechaza_un_factor_de_varias_columnas(self, panel_sintetico):
        panel, mercado = panel_sintetico
        with pytest.raises(ValueError, match="debe ser una serie"):
            rolling_market_correlation(panel, panel, window=VENTANA)

    def test_rechaza_un_factor_que_no_cubre_el_panel(self, panel_sintetico):
        """Un factor incompleto daria NaN en silencio; mejor que truene.

        Es el error facil de cometer al traer el factor de Databricks con un
        filtro de fechas distinto al del panel.
        """
        panel, mercado = panel_sintetico
        with pytest.raises(ValueError, match="no cubre"):
            rolling_market_correlation(panel, mercado.iloc[10:], window=VENTANA)


class TestDispersion:
    def test_la_amplitud_es_la_distancia_entre_p10_y_p90(self, panel_sintetico):
        panel, mercado = panel_sintetico
        d = correlation_dispersion(
            rolling_market_correlation(panel, mercado, window=VENTANA)
        )
        assert np.allclose(d["amplitud"], d["p90"] - d["p10"])

    def test_columnas_identicas_no_tienen_dispersion(self):
        """Si todas las acciones correlacionan igual, la dispersion es cero.

        Caso limite con respuesta exacta: separa medir dispersion de medir
        nivel. La media puede ser alta y la dispersion nula a la vez.
        """
        fechas = pd.bdate_range("2020-01-01", periods=50)
        iguales = pd.DataFrame(
            np.tile(np.linspace(0.2, 0.9, 50).reshape(-1, 1), (1, 8)), index=fechas
        )
        d = correlation_dispersion(iguales)
        assert np.allclose(d["desv_est"], 0.0)
        assert np.allclose(d["amplitud"], 0.0)
        assert np.allclose(d["media"], d["mediana"])

    def test_cuenta_las_acciones_disponibles(self):
        """``n_acciones`` existe para no confundir un panel que se vacia."""
        fechas = pd.bdate_range("2020-01-01", periods=3)
        con_huecos = pd.DataFrame(
            {"a": [0.5, 0.5, 0.5], "b": [0.7, np.nan, 0.7], "c": [np.nan, np.nan, 0.9]},
            index=fechas,
        )
        d = correlation_dispersion(con_huecos)
        assert list(d["n_acciones"]) == [2, 1, 3]

    def test_conserva_la_fecha_donde_solo_queda_una_accion(self):
        """Con una sola accion el promedio es valido y la dispersion no.

        La fila se conserva, con ``desv_est`` en NaN. Tirarla escondería que
        el panel se quedo sin datos justo ahi.
        """
        fechas = pd.bdate_range("2020-01-01", periods=2)
        casi_vacio = pd.DataFrame(
            {"a": [0.5, np.nan], "b": [0.9, np.nan], "c": [np.nan, 0.4]}, index=fechas
        )
        d = correlation_dispersion(casi_vacio)
        assert len(d) == 2
        assert d["media"].iloc[1] == pytest.approx(0.4)
        assert np.isnan(d["desv_est"].iloc[1])

    def test_descarta_las_fechas_sin_ninguna_accion(self):
        fechas = pd.bdate_range("2020-01-01", periods=3)
        vacio_en_medio = pd.DataFrame(
            {"a": [0.5, np.nan, 0.6], "b": [0.7, np.nan, 0.8]}, index=fechas
        )
        d = correlation_dispersion(vacio_en_medio)
        assert len(d) == 2
        assert fechas[1] not in d.index

    def test_detecta_que_la_amplitud_se_encoge(self):
        """El hallazgo que la funcion existe para medir.

        Se arma a proposito un periodo donde las acciones van de dispersas a
        sincronizadas. ``amplitud`` tiene que bajar aunque ``media`` suba.
        """
        dispersas = pd.DataFrame([np.linspace(0.0, 0.9, 20)])
        juntas = pd.DataFrame([np.linspace(0.80, 0.95, 20)])
        d = correlation_dispersion(pd.concat([dispersas, juntas], ignore_index=True))
        assert d["media"].iloc[1] > d["media"].iloc[0]
        assert d["amplitud"].iloc[1] < d["amplitud"].iloc[0]


class TestFisher:
    def test_es_la_inversa_de_la_tangente_hiperbolica(self):
        """``tanh(arctanh(r))`` tiene que devolver r."""
        r = pd.DataFrame({"a": [-0.9, -0.2, 0.0, 0.45, 0.95]})
        assert np.allclose(np.tanh(fisher_z(r)), r)

    def test_conserva_indice_y_columnas(self, panel_sintetico):
        panel, mercado = panel_sintetico
        rc = rolling_market_correlation(panel, mercado, window=VENTANA)
        z = fisher_z(rc)
        assert z.index.equals(rc.index)
        assert z.columns.equals(rc.columns)

    def test_no_produce_infinitos_en_la_correlacion_perfecta(self, panel_sintetico):
        """``arctanh(1)`` es infinito, y una accion puede replicar al mercado.

        El recorte evita que una sola columna perfecta contamine con inf
        todos los promedios de ``correlation_dispersion``.
        """
        panel, mercado = panel_sintetico
        z = fisher_z(rolling_market_correlation(panel, mercado, window=VENTANA))
        assert np.isfinite(z.to_numpy()).all()

    def test_quita_el_techo_que_comprime_la_dispersion(self):
        """La razon de ser de la funcion, como caso minimo.

        Dos fechas con la MISMA amplitud en r, una cerca del techo y otra en
        la zona media. En r se ven iguales; en z la de arriba se abre mas,
        que es justo el sesgo que ``correlation_dispersion`` no puede ver
        sola.
        """
        media_tabla = pd.DataFrame([np.linspace(0.30, 0.50, 20)])
        cerca_del_techo = pd.DataFrame([np.linspace(0.75, 0.95, 20)])
        juntas = pd.concat([media_tabla, cerca_del_techo], ignore_index=True)

        en_r = correlation_dispersion(juntas)
        assert en_r["amplitud"].iloc[0] == pytest.approx(en_r["amplitud"].iloc[1])

        en_z = correlation_dispersion(fisher_z(juntas))
        assert en_z["amplitud"].iloc[1] > en_z["amplitud"].iloc[0]
