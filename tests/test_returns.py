"""Pruebas de la transformacion de precios a rendimientos (``src/returns.py``).

Es el modulo del que cuelga todo lo demas: cada correlacion, beta y matriz de
covarianza del proyecto parte de un panel balanceado de rendimientos. Si una de
estas tres funciones se equivoca, el error se propaga sin avisar.

Mismo criterio que ``test_risk.py``: sin internet y sobre tablas diminutas
cuya respuesta se calcula a mano.

Correr con:

    SP500_SOLO_RESPALDO=1 ./.venv/bin/python -m pytest tests/test_returns.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.returns import build_balanced_panel, filter_by_history, to_log_returns


def _precios(**columnas: list[float]) -> pd.DataFrame:
    """Arma una tabla de precios con fechas habiles consecutivas."""
    n = len(next(iter(columnas.values())))
    fechas = pd.bdate_range("2024-01-01", periods=n, name="date")
    return pd.DataFrame(columnas, index=fechas)


NAN = np.nan


# ==========================================================================
# filter_by_history
# ==========================================================================
class TestFilterByHistory:
    def test_descarta_al_que_no_tiene_historia_suficiente(self):
        precios = _precios(
            A=[1.0] * 10,
            B=[NAN] * 5 + [1.0] * 5,  # empresa "reciente": 5 observaciones
        )
        filtrado, descartados = filter_by_history(precios, min_observations=8)

        assert list(filtrado.columns) == ["A"]
        assert descartados == ["B"]

    def test_el_minimo_es_inclusivo(self):
        """Con exactamente el minimo de observaciones, el ticker se queda."""
        precios = _precios(A=[1.0] * 8 + [NAN] * 2)
        filtrado, descartados = filter_by_history(precios, min_observations=8)

        assert list(filtrado.columns) == ["A"]
        assert descartados == []

    def test_cuenta_observaciones_y_no_filas(self):
        """Diez filas con cuatro huecos son seis observaciones, no diez."""
        precios = _precios(A=[1.0, NAN, 1.0, NAN, 1.0, NAN, 1.0, NAN, 1.0, 1.0])
        _, descartados = filter_by_history(precios, min_observations=7)

        assert descartados == ["A"]

    def test_los_descartados_salen_ordenados(self):
        precios = _precios(Z=[NAN] * 9 + [1.0], A=[NAN] * 9 + [1.0], M=[NAN] * 9 + [1.0])
        _, descartados = filter_by_history(precios, min_observations=5)

        assert descartados == ["A", "M", "Z"]

    def test_no_quita_filas(self):
        """Solo filtra columnas: las fechas se conservan intactas."""
        precios = _precios(A=[1.0] * 10, B=[NAN] * 5 + [1.0] * 5)
        filtrado, _ = filter_by_history(precios, min_observations=8)

        assert filtrado.index.equals(precios.index)

    def test_usa_el_minimo_de_config_si_no_se_indica(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_OBSERVATIONS", 8)
        precios = _precios(A=[1.0] * 10, B=[NAN] * 5 + [1.0] * 5)

        _, descartados = filter_by_history(precios)

        assert descartados == ["B"]

    def test_si_nadie_cumple_queda_una_tabla_sin_columnas(self):
        precios = _precios(A=[1.0] * 10, B=[1.0] * 10)
        filtrado, descartados = filter_by_history(precios, min_observations=100)

        assert filtrado.shape == (10, 0)
        assert descartados == ["A", "B"]


# ==========================================================================
# build_balanced_panel
# ==========================================================================
class TestBuildBalancedPanel:
    def test_el_panel_no_tiene_huecos(self):
        precios = _precios(
            A=[1.0, 2.0, NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            B=[1.0] * 10,
            C=[NAN] * 7 + [1.0] * 3,
        )
        panel, _ = build_balanced_panel(precios, min_observations=8)

        assert not panel.isna().any().any()

    def test_primero_se_filtran_tickers_y_despues_fechas(self):
        """El orden importa: es la razon de ser de la funcion.

        ``C`` es una empresa reciente: solo existe en los ultimos 3 de 10 dias.
        Si se borraran primero las fechas con huecos, ``C`` se llevaria 7 dias
        de historia de ``A`` y ``B``. Filtrando primero los tickers, ``C`` sale
        y las otras dos conservan los 10 dias.
        """
        precios = _precios(A=[1.0] * 10, B=[2.0] * 10, C=[NAN] * 7 + [3.0] * 3)
        panel, bitacora = build_balanced_panel(precios, min_observations=8)

        assert list(panel.columns) == ["A", "B"]
        assert len(panel) == 10
        assert bitacora["lista_descartados"] == ["C"]
        assert bitacora["filas_descartadas"] == 0

    def test_un_hueco_aislado_cuesta_la_fecha_y_no_el_ticker(self):
        """``D`` tiene 9 de 10 datos: pasa el filtro y solo pierde esa fecha."""
        precios = _precios(
            A=[1.0] * 10,
            D=[1.0, 1.0, 1.0, 1.0, NAN, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        panel, bitacora = build_balanced_panel(precios, min_observations=8)

        assert list(panel.columns) == ["A", "D"]
        assert len(panel) == 9
        assert precios.index[4] not in panel.index
        assert bitacora["filas_descartadas"] == 1

    def test_no_rellena_hacia_adelante(self):
        """Los valores que sobreviven son los originales, no inventados.

        Rellenar con ``ffill`` fabricaria dias con rendimiento de 0% que nunca
        ocurrieron y sesgaria las correlaciones hacia abajo.
        """
        precios = _precios(
            A=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            D=[1.0, 2.0, NAN, 4.0, 5.0, 6.0],
        )
        panel, _ = build_balanced_panel(precios, min_observations=5)

        esperado = precios.drop(index=precios.index[2])
        pd.testing.assert_frame_equal(panel, esperado, check_freq=False)

    def test_la_bitacora_cuadra(self):
        precios = _precios(
            A=[1.0] * 10,
            B=[2.0, 2.0, NAN, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            C=[NAN] * 7 + [3.0] * 3,
        )
        panel, b = build_balanced_panel(precios, min_observations=8)

        assert b["tickers_iniciales"] == 3
        assert b["tickers_descartados"] == len(b["lista_descartados"]) == 1
        assert b["tickers_finales"] == panel.shape[1] == 2
        assert b["tickers_iniciales"] - b["tickers_descartados"] == b["tickers_finales"]
        assert b["filas_iniciales"] - b["filas_descartadas"] == b["filas_finales"]
        assert b["filas_finales"] == len(panel) == 9

    def test_un_panel_ya_completo_queda_igual(self):
        precios = _precios(A=[1.0, 2.0, 3.0, 4.0], B=[4.0, 3.0, 2.0, 1.0])
        panel, bitacora = build_balanced_panel(precios, min_observations=4)

        pd.testing.assert_frame_equal(panel, precios, check_freq=False)
        assert bitacora["tickers_descartados"] == 0
        assert bitacora["filas_descartadas"] == 0

    def test_no_modifica_la_entrada(self):
        precios = _precios(A=[1.0] * 10, C=[NAN] * 7 + [3.0] * 3)
        copia = precios.copy()

        build_balanced_panel(precios, min_observations=8)

        pd.testing.assert_frame_equal(precios, copia)


# ==========================================================================
# to_log_returns
# ==========================================================================
class TestToLogReturns:
    def test_aplica_la_formula_ln_de_p_entre_p_anterior(self):
        precios = _precios(A=[100.0, 110.0, 121.0])
        rendimientos = to_log_returns(precios)

        esperado = [np.log(110 / 100), np.log(121 / 110)]
        assert np.allclose(rendimientos["A"], esperado)

    def test_la_primera_fila_se_elimina(self):
        """El primer dia no tiene un dia previo con que compararse."""
        precios = _precios(A=[100.0, 101.0, 102.0, 103.0])
        rendimientos = to_log_returns(precios)

        assert len(rendimientos) == len(precios) - 1
        assert rendimientos.index.equals(precios.index[1:])
        assert not rendimientos.isna().any().any()

    def test_conserva_las_columnas(self):
        precios = _precios(A=[1.0, 2.0, 3.0], B=[3.0, 2.0, 1.0])
        assert list(to_log_returns(precios).columns) == ["A", "B"]

    def test_los_logaritmos_se_suman_al_acumular(self):
        """La propiedad que justifica usar logaritmos.

        Una accion que sube 10% y luego baja 10% NO queda igual: 100 -> 110 ->
        99, perdio 1%. La suma de los rendimientos logaritmicos tiene que dar
        exactamente ese resultado de punta a punta.
        """
        precios = _precios(A=[100.0, 110.0, 99.0])
        rendimientos = to_log_returns(precios)

        de_punta_a_punta = np.log(99.0 / 100.0)
        assert rendimientos["A"].sum() == pytest.approx(de_punta_a_punta)

    def test_los_rendimientos_simples_no_se_suman(self):
        """El contraste: con rendimientos simples la misma suma da otra cosa.

        +10% y -10% suman 0%, pero la accion perdio 1%. Por eso el proyecto no
        promedia ni acumula rendimientos simples.
        """
        precios = _precios(A=[100.0, 110.0, 99.0])
        simples = precios.pct_change().iloc[1:]

        assert simples["A"].sum() == pytest.approx(0.0, abs=1e-12)
        assert np.log(99.0 / 100.0) != pytest.approx(0.0, abs=1e-3)

    def test_la_suma_es_el_rendimiento_total_en_una_serie_larga(self):
        """Lo mismo, sobre 250 dias de una caminata aleatoria y varias columnas."""
        rng = np.random.default_rng(7)
        pasos = np.exp(0.01 * rng.standard_normal((250, 3)))
        precios = pd.DataFrame(
            100 * np.cumprod(pasos, axis=0),
            index=pd.bdate_range("2023-01-02", periods=250, name="date"),
            columns=["A", "B", "C"],
        )
        rendimientos = to_log_returns(precios)

        esperado = np.log(precios.iloc[-1] / precios.iloc[0])
        assert np.allclose(rendimientos.sum(), esperado)

    def test_subir_y_bajar_lo_mismo_es_simetrico(self):
        """Duplicar el precio y volver a la mitad: +ln 2 y -ln 2, suma cero."""
        precios = _precios(A=[50.0, 100.0, 50.0])
        rendimientos = to_log_returns(precios)["A"]

        assert rendimientos.iloc[0] == pytest.approx(np.log(2))
        assert rendimientos.iloc[1] == pytest.approx(-np.log(2))
        assert rendimientos.sum() == pytest.approx(0.0)

    def test_un_precio_sin_cambio_da_cero(self):
        precios = _precios(A=[42.0, 42.0, 42.0])
        assert (to_log_returns(precios)["A"] == 0).all()

    def test_un_faltante_contagia_dos_dias_y_no_se_rellena(self):
        """Un hueco en el precio deja dos rendimientos sin definir.

        Falta el dia 3, asi que no se puede calcular el rendimiento del dia 3
        (no hay precio) ni el del dia 4 (no hay precio previo). La funcion no
        inventa nada: esos dos quedan como NaN.
        """
        precios = _precios(A=[100.0, 101.0, NAN, 103.0, 104.0])
        rendimientos = to_log_returns(precios)["A"]

        assert rendimientos.isna().tolist() == [False, True, True, False]

    def test_las_columnas_son_independientes(self):
        precios = _precios(A=[100.0, 105.0, 103.0], B=[10.0, 9.0, 12.0])
        juntas = to_log_returns(precios)
        solo_a = to_log_returns(precios[["A"]])

        pd.testing.assert_series_equal(juntas["A"], solo_a["A"])

    def test_no_modifica_la_entrada(self):
        precios = _precios(A=[100.0, 110.0, 99.0])
        copia = precios.copy()

        to_log_returns(precios)

        pd.testing.assert_frame_equal(precios, copia)
