"""Pruebas de la obtencion de datos (``src/data.py``).

``data.py`` es la puerta de entrada de todo el proyecto y habla con tres
servicios externos. Estas pruebas no tocan ninguno: Yahoo, Wikipedia y FRED se
sustituyen por respuestas falsas, asi que corren sin internet y dan siempre el
mismo resultado.

Lo que fijan es la parte que el proyecto promete en su docstring:

* la cadena **cache -> descarga -> respaldo -> error claro**,
* que la falta de una llave de API **no revienta** nada,
* que ``auto_adjust=True`` se pide siempre (sin eso un split parece una caida),
* y que Wikipedia y Yahoo escriben distinto los tickers con clase de accion.

``test_reproducibilidad.py`` ya prueba la bandera ``SP500_SOLO_RESPALDO`` cuando
no hay cache; aqui no se repite.

Correr con:

    SP500_SOLO_RESPALDO=1 ./.venv/bin/python -m pytest tests/test_data.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import requests

from src import config
from src import data as modulo_data


@pytest.fixture(autouse=True)
def sin_modo_respaldo(monkeypatch):
    """Aisla estas pruebas de la bandera ``SP500_SOLO_RESPALDO`` del entorno.

    Integracion continua la activa para todo; aqui hace falta que cada prueba
    decida por si misma si el modo esta encendido.
    """
    monkeypatch.setattr(modulo_data.config, "SOLO_RESPALDO", False)


class RespuestaFalsa:
    """Imita lo minimo de ``requests.Response`` que usa el modulo."""

    def __init__(self, texto: str = "", datos: dict | None = None,
                 error: Exception | None = None):
        self.text = texto
        self._datos = datos
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> dict:
        return self._datos


def _tabla_html(filas: list[tuple[str, str, str, str]], simbolo: str = "Symbol") -> str:
    """Tabla de constituyentes con el formato de las paginas de Wikipedia."""
    cabecera = (
        f"<thead><tr><th>{simbolo}</th><th>Security</th><th>GICS Sector</th>"
        "<th>GICS Sub-Industry</th></tr></thead>"
    )
    cuerpo = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in filas
    )
    return f"<table>{cabecera}<tbody>{cuerpo}</tbody></table>"


TABLA_IRRELEVANTE = (
    "<table><thead><tr><th>Fecha</th><th>Cambio</th></tr></thead>"
    "<tbody><tr><td>2024-01-01</td><td>nada</td></tr></tbody></table>"
)


# ==========================================================================
# to_yahoo_ticker
# ==========================================================================
class TestToYahooTicker:
    @pytest.mark.parametrize(
        "wikipedia, yahoo",
        [
            ("BRK.B", "BRK-B"),   # el caso que motivo la funcion
            ("BF.B", "BF-B"),
            ("AAPL", "AAPL"),     # sin punto no cambia
            ("brk.b", "BRK-B"),   # normaliza a mayusculas
            ("  msft  ", "MSFT"),  # y quita espacios
        ],
    )
    def test_traduce_el_formato_de_wikipedia_al_de_yahoo(self, wikipedia, yahoo):
        assert modulo_data.to_yahoo_ticker(wikipedia) == yahoo


# ==========================================================================
# _extract_close
# ==========================================================================
class TestExtractClose:
    FECHAS = pd.bdate_range("2024-01-01", periods=3)

    def _valores(self, columnas):
        return pd.DataFrame(np.arange(3 * len(columnas), dtype=float).reshape(3, -1),
                            index=self.FECHAS, columns=columnas)

    def test_campo_en_el_nivel_cero(self):
        """Varios tickers, columnas (campo, ticker): formato clasico de yfinance."""
        columnas = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
        bajado = self._valores(columnas)

        cierre = modulo_data._extract_close(bajado)

        assert list(cierre.columns) == ["AAA", "BBB"]
        assert np.array_equal(cierre.to_numpy(), bajado["Close"].to_numpy())

    def test_campo_en_el_nivel_uno(self):
        """La misma descarga con los niveles al reves, segun la version."""
        columnas = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close", "Open"]])
        bajado = self._valores(columnas)

        cierre = modulo_data._extract_close(bajado)

        assert list(cierre.columns) == ["AAA", "BBB"]
        assert np.array_equal(cierre.to_numpy(), bajado.xs("Close", axis=1, level=1).to_numpy())

    def test_un_solo_ticker_trae_columnas_de_un_nivel(self):
        bajado = self._valores(["Close", "Open"])

        cierre = modulo_data._extract_close(bajado)

        assert list(cierre.columns) == ["Close"]
        assert np.array_equal(cierre["Close"].to_numpy(), bajado["Close"].to_numpy())

    def test_sin_columna_close_el_error_lo_dice(self):
        columnas = pd.MultiIndex.from_product([["Open", "High"], ["AAA"]])
        with pytest.raises(KeyError, match="Close"):
            modulo_data._extract_close(self._valores(columnas))


# ==========================================================================
# _load_with_fallback: cache -> descarga -> respaldo -> error
# ==========================================================================
class TestCadenaDeRespaldo:
    DESCARGADO = pd.DataFrame({"x": [1, 2, 3]})
    EN_CACHE = pd.DataFrame({"x": [10, 20, 30]})
    EN_RESPALDO = pd.DataFrame({"x": [100, 200, 300]})

    @staticmethod
    def _no_debe_llamarse():
        raise AssertionError("se intento descargar teniendo el dato a mano")

    def test_si_hay_cache_no_se_descarga(self, tmp_path):
        cache = tmp_path / "cache.parquet"
        self.EN_CACHE.to_parquet(cache)

        frame = modulo_data._load_with_fallback(
            cache=cache, fallback=tmp_path / "no.parquet",
            download=self._no_debe_llamarse, label="prueba",
        )

        pd.testing.assert_frame_equal(frame, self.EN_CACHE)

    def test_force_refresh_ignora_el_cache_y_lo_reescribe(self, tmp_path):
        cache = tmp_path / "cache.parquet"
        self.EN_CACHE.to_parquet(cache)

        frame = modulo_data._load_with_fallback(
            cache=cache, fallback=tmp_path / "no.parquet",
            download=lambda: self.DESCARGADO, label="prueba", force_refresh=True,
        )

        pd.testing.assert_frame_equal(frame, self.DESCARGADO)
        pd.testing.assert_frame_equal(pd.read_parquet(cache), self.DESCARGADO)

    def test_una_descarga_exitosa_se_guarda_en_el_cache(self, tmp_path):
        cache = tmp_path / "cache.parquet"
        assert not cache.exists()

        modulo_data._load_with_fallback(
            cache=cache, fallback=tmp_path / "no.parquet",
            download=lambda: self.DESCARGADO, label="prueba",
        )

        assert cache.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(cache), self.DESCARGADO)

    def test_si_la_descarga_falla_usa_el_respaldo_y_avisa(self, tmp_path):
        respaldo = tmp_path / "respaldo.parquet"
        self.EN_RESPALDO.to_parquet(respaldo)

        def descarga_caida():
            raise ConnectionError("sin red")

        with pytest.warns(UserWarning, match="respaldo"):
            frame = modulo_data._load_with_fallback(
                cache=tmp_path / "no_existe.parquet", fallback=respaldo,
                download=descarga_caida, label="prueba",
            )

        pd.testing.assert_frame_equal(frame, self.EN_RESPALDO)

    def test_si_no_hay_nada_el_error_conserva_la_causa(self, tmp_path):
        original = ConnectionError("sin red")

        def descarga_caida():
            raise original

        with pytest.raises(RuntimeError, match="no hay respaldo") as info:
            modulo_data._load_with_fallback(
                cache=tmp_path / "no_existe.parquet",
                fallback=tmp_path / "tampoco.parquet",
                download=descarga_caida, label="prueba",
            )

        assert info.value.__cause__ is original

    def test_en_modo_respaldo_el_cache_gana_al_respaldo(self, monkeypatch, tmp_path):
        """El docstring lo promete: con la bandera activa, el cache va primero."""
        monkeypatch.setattr(modulo_data.config, "SOLO_RESPALDO", True)
        cache = tmp_path / "cache.parquet"
        respaldo = tmp_path / "respaldo.parquet"
        self.EN_CACHE.to_parquet(cache)
        self.EN_RESPALDO.to_parquet(respaldo)

        frame = modulo_data._load_with_fallback(
            cache=cache, fallback=respaldo,
            download=self._no_debe_llamarse, label="prueba",
        )

        pd.testing.assert_frame_equal(frame, self.EN_CACHE)


# ==========================================================================
# get_fred_series: la llave es opcional
# ==========================================================================
class TestFred:
    RESPALDO = pd.DataFrame(
        {"vix": [20.0, 21.0], "treasury_10y": [1.5, 1.6]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]).rename("date"),
    )

    @pytest.fixture
    def rutas(self, monkeypatch, tmp_path):
        """Apunta cache y respaldo de FRED a una carpeta temporal."""
        monkeypatch.setattr(modulo_data.config, "MACRO_CACHE", tmp_path / "cache.parquet")
        monkeypatch.setattr(modulo_data.config, "MACRO_FALLBACK", tmp_path / "respaldo.parquet")
        monkeypatch.setattr(modulo_data.config, "FRED_API_KEY", "")
        return tmp_path

    def test_sin_llave_usa_el_respaldo_en_lugar_de_fallar(self, rutas):
        self.RESPALDO.to_parquet(rutas / "respaldo.parquet")

        with pytest.warns(UserWarning, match="FRED_API_KEY"):
            frame = modulo_data.get_fred_series()

        pd.testing.assert_frame_equal(frame, self.RESPALDO)

    def test_sin_llave_ni_respaldo_devuelve_una_tabla_vacia(self, rutas):
        with pytest.warns(UserWarning, match="tabla vac"):
            frame = modulo_data.get_fred_series()

        assert isinstance(frame, pd.DataFrame)
        assert frame.empty

    def test_si_hay_cache_no_avisa_ni_pide_nada(self, rutas):
        self.RESPALDO.to_parquet(rutas / "cache.parquet")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            frame = modulo_data.get_fred_series()

        pd.testing.assert_frame_equal(frame, self.RESPALDO)

    def test_con_llave_pide_las_dos_series_y_convierte_el_punto_en_nan(
        self, rutas, monkeypatch
    ):
        """FRED marca los faltantes con un punto ".", no con un nulo."""
        monkeypatch.setattr(modulo_data.config, "FRED_API_KEY", "llave-de-prueba")
        pedidos = []

        def get_falso(url, params=None, **kwargs):
            pedidos.append(params)
            return RespuestaFalsa(datos={"observations": [
                {"date": "2024-01-02", "value": "13.5"},
                {"date": "2024-01-03", "value": "."},
            ]})

        monkeypatch.setattr(modulo_data.requests, "get", get_falso)

        macro = modulo_data.get_fred_series()

        assert {p["series_id"] for p in pedidos} == {"VIXCLS", "DGS10"}
        assert all(p["api_key"] == "llave-de-prueba" for p in pedidos)
        assert list(macro.columns) == ["vix", "treasury_10y"]
        assert macro.index.name == "date"
        assert macro["vix"].iloc[0] == 13.5
        assert macro["vix"].isna().tolist() == [False, True]
        assert (rutas / "cache.parquet").exists()


# ==========================================================================
# Scraping de Wikipedia
# ==========================================================================
class TestDescargaDeIndices:
    URL = "https://ejemplo.test/lista"

    @staticmethod
    def _servir(monkeypatch, respuesta, llamadas=None):
        def get_falso(url, headers=None, timeout=None, **kwargs):
            if llamadas is not None:
                llamadas.append({"url": url, "headers": headers, "timeout": timeout})
            return respuesta

        monkeypatch.setattr(modulo_data.requests, "get", get_falso)

    def test_encuentra_la_tabla_aunque_no_sea_la_primera(self, monkeypatch):
        """Se busca por columnas, no por posicion: Wikipedia puede reordenar."""
        html = TABLA_IRRELEVANTE + _tabla_html([("AAA", "Alfa", "Energy", "Oil")])
        self._servir(monkeypatch, RespuestaFalsa(texto=html))

        frame = modulo_data._download_index_members(self.URL, "large")

        assert frame["ticker"].tolist() == ["AAA"]
        assert frame["sector"].tolist() == ["Energy"]

    def test_acepta_la_columna_ticker_en_lugar_de_symbol(self, monkeypatch):
        html = _tabla_html([("BBB", "Beta", "Materials", "Gold")], simbolo="Ticker")
        self._servir(monkeypatch, RespuestaFalsa(texto=html))

        frame = modulo_data._download_index_members(self.URL, "mid")

        assert frame["ticker"].tolist() == ["BBB"]

    def test_traduce_tickers_y_agrega_el_nivel_de_capitalizacion(self, monkeypatch):
        html = _tabla_html([
            ("BRK.B", "Berkshire", "Financials", "Insurance"),
            ("AAPL", "Apple", "Information Technology", "Hardware"),
        ])
        self._servir(monkeypatch, RespuestaFalsa(texto=html))

        frame = modulo_data._download_index_members(self.URL, "small")

        assert frame["ticker"].tolist() == ["BRK-B", "AAPL"]
        assert (frame["cap_tier"] == "small").all()
        assert list(frame.columns) == ["ticker", "company", "sector", "sub_industry", "cap_tier"]

    def test_se_presenta_con_un_user_agent(self, monkeypatch):
        """Wikipedia rechaza las peticiones sin User-Agent de navegador."""
        llamadas = []
        html = _tabla_html([("AAA", "Alfa", "Energy", "Oil")])
        self._servir(monkeypatch, RespuestaFalsa(texto=html), llamadas)

        modulo_data._download_index_members(self.URL, "large")

        assert llamadas[0]["url"] == self.URL
        assert "User-Agent" in llamadas[0]["headers"]
        assert llamadas[0]["timeout"] is not None  # nunca se queda colgado

    def test_si_no_hay_tabla_de_constituyentes_el_error_lo_explica(self, monkeypatch):
        self._servir(monkeypatch, RespuestaFalsa(texto=TABLA_IRRELEVANTE))

        with pytest.raises(RuntimeError, match="No se encontro la tabla"):
            modulo_data._download_index_members(self.URL, "large")

    def test_un_error_http_no_se_traga(self, monkeypatch):
        error = requests.HTTPError("403 Forbidden")
        self._servir(monkeypatch, RespuestaFalsa(error=error))

        with pytest.raises(requests.HTTPError):
            modulo_data._download_index_members(self.URL, "large")


# ==========================================================================
# S&P Composite 1500: union de los tres indices
# ==========================================================================
class TestUniversoSp1500:
    @staticmethod
    def _miembros(filas, cap_tier):
        return pd.DataFrame({
            "ticker": [f[0] for f in filas],
            "company": [f"Empresa {f[0]}" for f in filas],
            "sector": "Energy",
            "sub_industry": "Oil",
            "cap_tier": cap_tier,
        })

    @pytest.fixture
    def union(self, monkeypatch):
        """Une tres indices falsos donde DUP1 y DUP2 aparecen en mas de uno."""
        por_nivel = {
            "large": [("AAA",), ("DUP1",)],
            "mid": [("BBB",), ("DUP1",), ("DUP2",)],
            "small": [("CCC",), ("DUP1",), ("DUP2",)],
        }
        monkeypatch.setattr(
            modulo_data, "_download_index_members",
            lambda url, cap_tier: self._miembros(por_nivel[cap_tier], cap_tier),
        )
        return modulo_data._download_sp1500_constituents()

    def test_un_ticker_en_varios_indices_queda_una_sola_vez(self, union):
        assert union["ticker"].is_unique
        assert union["ticker"].tolist() == ["AAA", "BBB", "CCC", "DUP1", "DUP2"]

    def test_gana_el_indice_de_mayor_capitalizacion(self, union):
        niveles = union.set_index("ticker")["cap_tier"]

        assert niveles["DUP1"] == "large"  # estaba en los tres
        assert niveles["DUP2"] == "mid"    # estaba en mid y small

    def test_no_deja_columnas_auxiliares_y_el_indice_es_limpio(self, union):
        assert list(union.columns) == ["ticker", "company", "sector", "sub_industry", "cap_tier"]
        assert union.index.tolist() == list(range(len(union)))

    def test_el_nivel_es_de_tipo_texto(self, union):
        assert str(union["cap_tier"].dtype) == "string"


# ==========================================================================
# Precios de Yahoo Finance
# ==========================================================================
class TestPrecios:
    @staticmethod
    def _descarga_falsa(llamadas):
        """Devuelve lo que ``yf.download`` regresaria: fechas desordenadas."""
        fechas = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
        columnas = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
        valores = np.arange(12, dtype=float).reshape(3, 4)

        def falsa(**kwargs):
            llamadas.append(kwargs)
            return pd.DataFrame(valores, index=fechas, columns=columnas)

        return falsa

    def test_siempre_pide_precios_ajustados(self, monkeypatch, tmp_path):
        """Sin ``auto_adjust`` un split 4:1 se veria como una caida de -75%."""
        llamadas = []
        monkeypatch.setattr(modulo_data.yf, "download", self._descarga_falsa(llamadas))

        modulo_data.get_prices(
            ["AAA", "BBB"], cache=tmp_path / "c.parquet", fallback=tmp_path / "r.parquet"
        )

        assert llamadas[0]["auto_adjust"] is True

    def test_devuelve_formato_ancho_ordenado_por_fecha(self, monkeypatch, tmp_path):
        monkeypatch.setattr(modulo_data.yf, "download", self._descarga_falsa([]))

        precios = modulo_data.get_prices(
            ["AAA", "BBB"], cache=tmp_path / "c.parquet", fallback=tmp_path / "r.parquet"
        )

        assert list(precios.columns) == ["AAA", "BBB"]
        assert precios.index.name == "date"
        assert precios.index.is_monotonic_increasing

    def test_usa_las_fechas_de_config_si_no_se_indican(self, monkeypatch, tmp_path):
        llamadas = []
        monkeypatch.setattr(modulo_data.yf, "download", self._descarga_falsa(llamadas))

        modulo_data.get_prices(
            ["AAA"], cache=tmp_path / "c.parquet", fallback=tmp_path / "r.parquet"
        )

        assert llamadas[0]["start"] == config.START_DATE
        assert llamadas[0]["end"] == config.END_DATE

    def test_la_segunda_llamada_sale_del_cache(self, monkeypatch, tmp_path):
        llamadas = []
        monkeypatch.setattr(modulo_data.yf, "download", self._descarga_falsa(llamadas))
        rutas = {"cache": tmp_path / "c.parquet", "fallback": tmp_path / "r.parquet"}

        modulo_data.get_prices(["AAA", "BBB"], **rutas)
        modulo_data.get_prices(["AAA", "BBB"], **rutas)

        assert len(llamadas) == 1

    def test_una_descarga_vacia_cae_al_respaldo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(modulo_data.yf, "download", lambda **kw: pd.DataFrame())
        respaldo = tmp_path / "r.parquet"
        pd.DataFrame({"ZZZ": [1.0, 2.0]}).to_parquet(respaldo)

        with pytest.warns(UserWarning, match="tabla vac"):
            precios = modulo_data.get_prices(
                ["AAA"], cache=tmp_path / "c.parquet", fallback=respaldo
            )

        assert list(precios.columns) == ["ZZZ"]

    def test_descarga_vacia_y_sin_respaldo_es_un_error_claro(self, monkeypatch, tmp_path):
        monkeypatch.setattr(modulo_data.yf, "download", lambda **kw: pd.DataFrame())

        with pytest.raises(RuntimeError, match="no hay respaldo"):
            modulo_data.get_prices(
                ["AAA"], cache=tmp_path / "c.parquet", fallback=tmp_path / "r.parquet"
            )


# ==========================================================================
# Los respaldos que viajan con el repositorio
# ==========================================================================
class TestRespaldosDelRepositorio:
    """Con la bandera activa y sin cache, la tabla sale del repositorio.

    Es lo que garantiza que quien clone el proyecto pueda ejecutarlo aunque
    Wikipedia o Yahoo no respondan ese dia.
    """

    @pytest.fixture(autouse=True)
    def solo_respaldo_sin_cache(self, sin_modo_respaldo, monkeypatch, tmp_path):
        # Depende de ``sin_modo_respaldo`` para que el orden sea siempre el
        # mismo: primero se apaga la bandera y despues esta prueba la enciende.
        monkeypatch.setattr(modulo_data.config, "SOLO_RESPALDO", True)
        monkeypatch.setattr(modulo_data.config, "CONSTITUENTS_CACHE", tmp_path / "a.parquet")
        monkeypatch.setattr(modulo_data.config, "SP1500_CONSTITUENTS_CACHE", tmp_path / "b.parquet")

    def test_sp500_trae_las_cuatro_columnas(self):
        empresas = modulo_data.get_sp500_constituents()

        assert list(empresas.columns) == ["ticker", "company", "sector", "sub_industry"]
        assert empresas["ticker"].is_unique
        assert len(empresas) >= 500

    def test_sp1500_trae_los_tres_niveles(self):
        empresas = modulo_data.get_sp1500_constituents()

        assert set(empresas["cap_tier"]) == {"large", "mid", "small"}
        assert empresas["ticker"].is_unique
        assert len(empresas) >= 1500

    def test_los_tickers_ya_vienen_en_formato_yahoo(self):
        """Ningun ticker conserva el punto de Wikipedia: BRK.B debe ser BRK-B."""
        for empresas in (modulo_data.get_sp500_constituents(),
                         modulo_data.get_sp1500_constituents()):
            assert not empresas["ticker"].str.contains(".", regex=False).any()
            assert "BRK-B" in set(empresas["ticker"])
