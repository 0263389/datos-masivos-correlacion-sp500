"""Fija las cifras publicadas del proyecto contra el codigo que las produce.

Por que existe
--------------
El resto de las pruebas verifica que cada funcion hace lo que promete. Ninguna
verifica que **el proyecto completo sigue dando los numeros que el README y el
terminal publican**. Son cosas distintas: todas las piezas pueden estar
correctas y aun asi una cifra citada puede haber dejado de ser cierta, porque
cambio un umbral de configuracion, el orden de dos pasos, o el criterio de
exclusion.

Estas pruebas recorren la cadena entera -extraccion, panel balanceado,
rendimientos, exclusion de datos corruptos, correlacion, agregacion- y comparan
el resultado contra la cifra publicada. Si alguna cambia, hay que decidir a
conciencia si el numero nuevo es el correcto y actualizar el README, en lugar
de enterarse cuando alguien lo pregunte en una defensa.

Las tolerancias son estrictas a proposito. Una diferencia en el cuarto decimal
no es ruido de punto flotante: significa que el conjunto de activos o el
periodo cambiaron.

De donde salen los datos
------------------------
Siempre del respaldo versionado en ``data/processed/``, nunca de una descarga
en vivo ni del cache local. No es un detalle: las cifras de abajo describen un
conjunto **congelado**, y la composicion del S&P 500 cambia. Al probarlo sobre
un clon limpio con internet, Wikipedia devolvio siete empresas que entraron al
indice y siete que salieron desde que se congelo el panel; los precios seguian
viniendo del respaldo, asi que siete tickers del panel se quedaron sin sector y
los pares clasificados bajaron de 104,653 a 102,831.

Esa es la falla que estas pruebas tienen que descartar, no producir. Por eso el
fixture ``datos_congelados`` enciende ``SOLO_RESPALDO`` y ademas apunta los
caches a una ruta inexistente: sin lo segundo el cache ganaria, y en una maquina
que alguna vez descargo en vivo el resultado dependeria de cuando lo hizo.

Se omiten solas si ``data/`` no esta disponible, de modo que quien clone el
repositorio sin los parquet pueda correr el resto de la bateria sin fallos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analysis, config, data, quality, returns

HAY_DATOS = config.PRICES_CACHE.exists() or config.PRICES_FALLBACK.exists()
sin_datos = pytest.mark.skipif(
    not HAY_DATOS, reason="requiere los parquet de data/, que no se versionan completos"
)

# Cifras publicadas en el README, en el terminal y en la defensa del
# Entregable 1. Cambiar cualquiera exige actualizar tambien esos tres lugares.
TICKERS_FINALES = 458
PARES = 104_653
CORR_MISMO_SECTOR = 0.4404
CORR_DISTINTO_SECTOR = 0.3164
EXCLUIDOS_POR_CORRUPCION = ["MNST"]


@pytest.fixture(scope="module", autouse=True)
def datos_congelados(tmp_path_factory):
    """Obliga a leer el respaldo del repositorio y nada mas.

    ``SOLO_RESPALDO`` por si solo no basta: ``_load_with_fallback`` consulta el
    cache antes que la bandera -decision deliberada, probada en
    ``test_data.py::test_en_modo_respaldo_el_cache_gana_al_respaldo``-, asi que
    tambien se apuntan los caches a un directorio vacio. El resultado es el
    mismo en integracion continua, en un clon recien hecho y en la maquina de
    quien descargo precios el mes pasado.
    """
    vacio = tmp_path_factory.mktemp("sin_cache")
    parche = pytest.MonkeyPatch()
    parche.setattr(config, "SOLO_RESPALDO", True)
    parche.setattr(config, "PRICES_CACHE", vacio / "precios.parquet")
    parche.setattr(config, "CONSTITUENTS_CACHE", vacio / "constituyentes.parquet")
    yield
    parche.undo()


@pytest.fixture(scope="module")
def cadena():
    """Recorre el pipeline del Entregable 1 tal como lo hace el notebook."""
    constituyentes = data.get_sp500_constituents()
    precios = data.get_prices([])

    panel, bitacora = returns.build_balanced_panel(precios)
    rendimientos = returns.to_log_returns(panel)

    sospechosos = quality.flag_suspicious_tickers(rendimientos)
    excluidos = sospechosos[sospechosos["sospechoso"]].index.tolist()
    rendimientos = rendimientos.drop(columns=excluidos)

    pares = analysis.pair_table(analysis.correlation_matrix(rendimientos),
                                constituyentes)
    return {
        "panel": panel,
        "rendimientos": rendimientos,
        "excluidos": excluidos,
        "pares": pares,
        "comparacion": analysis.sector_comparison(pares),
        "bitacora": bitacora,
    }


@sin_datos
class TestCifrasDelEntregable1:
    def test_el_universo_final_es_el_publicado(self, cadena):
        assert cadena["rendimientos"].shape[1] == TICKERS_FINALES

    def test_se_excluye_exactamente_el_ticker_corrupto_conocido(self, cadena):
        """MNST tenia el split aplicado de forma inconsistente.

        La prueba fija tambien que **solo** se excluya ese: el criterio por
        frecuencia de eventos extremos llego a marcar acciones cuyos
        movimientos eran reales, y corregir eso fue uno de los hallazgos del
        proyecto. Si la lista crece, el criterio volvio a ser demasiado
        agresivo.
        """
        assert cadena["excluidos"] == EXCLUIDOS_POR_CORRUPCION

    def test_el_numero_de_pares_es_el_publicado(self, cadena):
        assert len(cadena["pares"]) == PARES
        # Y es el que corresponde al triangulo superior, sin la diagonal.
        n = cadena["rendimientos"].shape[1]
        assert len(cadena["pares"]) == n * (n - 1) // 2

    def test_la_correlacion_por_sector_es_la_publicada(self, cadena):
        comp = cadena["comparacion"]
        assert comp.loc["Mismo sector", "media"] == pytest.approx(
            CORR_MISMO_SECTOR, abs=5e-5)
        assert comp.loc["Distinto sector", "media"] == pytest.approx(
            CORR_DISTINTO_SECTOR, abs=5e-5)

    def test_la_razon_entre_sectores_es_la_que_se_defiende(self, cadena):
        """1.39x es la cifra que resume el Entregable 1."""
        comp = cadena["comparacion"]
        razon = comp.loc["Mismo sector", "media"] / comp.loc["Distinto sector", "media"]
        assert razon == pytest.approx(1.39, abs=0.005)

    def test_los_dos_grupos_cubren_todos_los_pares(self, cadena):
        assert int(cadena["comparacion"]["n_pares"].sum()) == PARES


@sin_datos
class TestIntegridadDelPanel:
    def test_el_panel_no_tiene_huecos(self, cadena):
        """Un panel balanceado sin huecos es el supuesto de todo lo demas."""
        assert not cadena["panel"].isna().any().any()

    def test_no_se_rellenaron_precios(self, cadena):
        """Rellenar inventaria sesiones de rendimiento cero.

        Se detecta por su rastro: un precio repetido exactamente produce un
        rendimiento de cero. Unos pocos ceros son normales en acciones poco
        liquidas; una proporcion alta delataria imputacion.
        """
        ceros = (cadena["rendimientos"] == 0).to_numpy().mean()
        assert ceros < 0.01, f"{ceros:.2%} de rendimientos exactamente cero"

    def test_las_fechas_estan_ordenadas_y_sin_repetir(self, cadena):
        idx = cadena["panel"].index
        assert idx.is_monotonic_increasing
        assert idx.is_unique

    def test_la_correlacion_esta_en_el_rango_valido(self, cadena):
        corr = cadena["pares"]["corr"]
        assert corr.between(-1, 1).all()
        assert corr.notna().all()

    def test_el_umbral_de_historia_es_el_medido(self):
        """2,900 no es un numero redondo elegido a ojo: se midio el intercambio
        entre cuantas empresas se conservan y cuanta historia queda.
        """
        assert config.MIN_OBSERVATIONS == 2900


@pytest.fixture(scope="module")
def mercado_garch(cadena):
    """Mercado equiponderado del panel final, con su GARCH ya ajustado."""
    from src.cofactors import mercado
    from src.volatility import ajustar_garch, volatilidad_condicional

    rend = cadena["rendimientos"]
    m = mercado(rend)
    par = ajustar_garch(m.to_numpy(dtype=np.float64))
    vol = pd.Series(volatilidad_condicional(m.to_numpy(dtype=np.float64), par),
                    index=rend.index)
    return m, par, vol


@sin_datos
class TestCifrasDelEntregable3:
    """Las que el terminal publica en su portada."""

    def test_el_garch_del_mercado_detecta_agrupamiento(self, mercado_garch):
        _, par, _ = mercado_garch
        assert par["hay_agrupamiento"]
        assert 0.90 < par["persistencia"] < 1.0

    def test_el_maximo_de_volatilidad_cae_en_marzo_de_2020(self, mercado_garch):
        """La afirmacion central del Entregable 3, y la mas facil de verificar.

        El modelo no conoce ningun calendario de eventos: si el maximo deja de
        caer en el desplome del COVID, algo se rompio en la cadena.
        """
        _, _, vol = mercado_garch
        maximo = vol.idxmax()
        assert maximo.year == 2020 and maximo.month == 3, f"el maximo cae en {maximo.date()}"

    def test_la_volatilidad_en_tension_duplica_la_de_calma(self, mercado_garch):
        from src.cofactors import tension_por_volatilidad

        m, par, _ = mercado_garch
        tension = tension_por_volatilidad(m, parametros=par)
        assert tension.mean() == pytest.approx(0.20, abs=0.01)
        assert m[tension].std() > 2 * m[~tension].std()


@sin_datos
class TestModoSinRed:
    """La bandera que vuelve deterministas estas pruebas.

    Sin ella, una maquina sin cache -la de integracion continua, o la de un
    companiero que acaba de clonar- descargaria precios frescos y obtendria un
    panel distinto al que las cifras publicadas describen. Las pruebas de
    arriba fallarian, pero por la razon equivocada.
    """

    def test_la_bandera_se_lee_del_entorno(self, monkeypatch):
        import importlib

        from src import config as cfg

        for valor, esperado in [("1", True), ("true", True), ("SI", True),
                                ("0", False), ("", False), ("no", False)]:
            monkeypatch.setenv("SP500_SOLO_RESPALDO", valor)
            assert importlib.reload(cfg).SOLO_RESPALDO is esperado, f"con {valor!r}"

        monkeypatch.delenv("SP500_SOLO_RESPALDO", raising=False)
        importlib.reload(cfg)

    def test_sin_cache_usa_el_respaldo_y_no_descarga(self, monkeypatch, tmp_path):
        """Prueba central: con la bandera activa, la descarga ni se intenta.

        Se simula una maquina recien clonada apuntando el cache a una ruta que
        no existe, y se sustituye la descarga por una funcion que falla si
        alguien la llama.
        """
        from src import data as modulo_data

        def no_debe_llamarse():
            raise AssertionError("se intentó descargar teniendo la bandera activa")

        monkeypatch.setattr(modulo_data.config, "SOLO_RESPALDO", True)
        frame = modulo_data._load_with_fallback(
            cache=tmp_path / "no_existe.parquet",
            fallback=config.PRICES_FALLBACK,
            download=no_debe_llamarse,
            label="prueba",
        )
        assert not frame.empty
        assert frame.shape[1] > 400

    def test_sin_respaldo_el_error_dice_que_hacer(self, monkeypatch, tmp_path):
        from src import data as modulo_data

        monkeypatch.setattr(modulo_data.config, "SOLO_RESPALDO", True)
        with pytest.raises(RuntimeError, match="SP500_SOLO_RESPALDO"):
            modulo_data._load_with_fallback(
                cache=tmp_path / "no_existe.parquet",
                fallback=tmp_path / "tampoco_existe.parquet",
                download=lambda: None,
                label="prueba",
            )
