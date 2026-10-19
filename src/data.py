"""Obtencion de datos desde las tres fuentes del proyecto.

Fuentes
-------
1. **Wikipedia** (scraping)  -> lista de constituyentes del S&P 500 con su
   sector GICS. Es lo que nos permite *interpretar* la matriz de correlacion.
2. **Yahoo Finance** (libreria yfinance) -> precios diarios ajustados.
3. **FRED** (API con llave)  -> series macro (VIX, tasa a 10 anios).

Cadena de respaldo
------------------
Toda funcion de descarga intenta lo mismo, en este orden:

    1. cache local en data/raw/   (rapido; ignorado por git)
    2. descarga en vivo           (la fuente real)
    3. respaldo en data/processed/ (viaja con el repositorio)
    4. error con mensaje claro

El punto 3 es el que hace reproducible el proyecto. Quien clone el repositorio
recibe ``data/raw/`` vacio: si la fuente esta caida ese dia, sin respaldo el
notebook seria inejecutable y la rubrica lo penaliza.

**Ninguna funcion de este modulo lanza una excepcion cuando falta una llave de
API.** Avisa y continua. El profesor que revise el proyecto no tiene por que
registrarse en FRED para poder ver los resultados.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src import config

# User-Agent generico: Wikipedia rechaza las peticiones de urllib por defecto,
# que es justo lo que usa pandas.read_html si le pasas una URL directamente.
_BROWSER_UA = {"User-Agent": "Mozilla/5.0 (compatible; proyecto-academico/1.0)"}


# ==========================================================================
# Utilidad compartida: cache -> descarga -> respaldo
# ==========================================================================
def _load_with_fallback(
    cache: Path,
    fallback: Path,
    download: Callable[[], pd.DataFrame],
    label: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Aplica la cadena cache -> descarga -> respaldo para una fuente."""
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)

    try:
        frame = download()
        frame.to_parquet(cache)
        return frame
    except Exception as error:  # noqa: BLE001 - queremos degradar, no reventar
        if fallback.exists():
            warnings.warn(
                f"[{label}] No se pudo descargar en vivo ({type(error).__name__}: "
                f"{error}). Se usa el respaldo incluido en el repositorio: "
                f"{fallback.name}",
                stacklevel=2,
            )
            return pd.read_parquet(fallback)
        raise RuntimeError(
            f"[{label}] Fallo la descarga y no hay respaldo disponible en "
            f"{fallback}. Error original: {error}"
        ) from error


# ==========================================================================
# 1. Constituyentes del S&P 500 (scraping de Wikipedia)
# ==========================================================================
def to_yahoo_ticker(symbol: str) -> str:
    """Traduce un simbolo estilo Wikipedia al formato de Yahoo Finance.

    Wikipedia usa punto para las clases de acciones (``BRK.B``, ``BF.B``);
    Yahoo usa guion (``BRK-B``, ``BF-B``). Sin esta traduccion, esos tickers
    regresan vacios de Yahoo **sin lanzar ningun error**, y simplemente
    desaparecen del analisis sin que nadie se entere.
    """
    return symbol.strip().upper().replace(".", "-")


def _download_constituents() -> pd.DataFrame:
    response = requests.get(config.SP500_WIKI_URL, headers=_BROWSER_UA, timeout=30)
    response.raise_for_status()

    # La primera tabla de la pagina es la de constituyentes actuales.
    raw = pd.read_html(StringIO(response.text))[0]

    return (
        pd.DataFrame(
            {
                "ticker": raw["Symbol"].map(to_yahoo_ticker),
                "company": raw["Security"].astype("string"),
                "sector": raw["GICS Sector"].astype("string"),
                "sub_industry": raw["GICS Sub-Industry"].astype("string"),
            }
        )
        .drop_duplicates(subset="ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """Descarga la tabla de empresas del S&P 500 con su sector GICS.

    Returns
    -------
    DataFrame con columnas: ticker, company, sector, sub_industry.
    """
    return _load_with_fallback(
        cache=config.CONSTITUENTS_CACHE,
        fallback=config.CONSTITUENTS_FALLBACK,
        download=_download_constituents,
        label="Wikipedia / constituyentes",
        force_refresh=force_refresh,
    )


# ==========================================================================
# 2. Precios diarios ajustados (Yahoo Finance)
# ==========================================================================
def _extract_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Saca el precio de cierre del resultado de ``yf.download``.

    yfinance devuelve columnas de dos niveles (campo, ticker) cuando se piden
    varios tickers, y de un solo nivel cuando se pide uno. Esta funcion
    normaliza ambos casos para que el resto del codigo no tenga que saberlo.
    """
    if isinstance(downloaded.columns, pd.MultiIndex):
        # El nivel que contiene "Close" puede ser el 0 o el 1 segun la version.
        for level in range(downloaded.columns.nlevels):
            if "Close" in downloaded.columns.get_level_values(level):
                return downloaded.xs("Close", axis=1, level=level)
        raise KeyError("No se encontro la columna 'Close' en la descarga.")
    return downloaded[["Close"]]


def get_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    force_refresh: bool = False,
    cache: Path | None = None,
    fallback: Path | None = None,
) -> pd.DataFrame:
    """Descarga precios de cierre **ajustados** en formato ancho.

    ``auto_adjust=True`` es obligatorio: devuelve precios corregidos por splits
    y dividendos. Con el cierre crudo, un split 4:1 se veria como una caida de
    -75% en un dia y contaminaria toda la matriz de correlacion.

    Returns
    -------
    DataFrame indexado por fecha, una columna por ticker.
    """

    def download() -> pd.DataFrame:
        downloaded = yf.download(
            tickers=tickers,
            start=start or config.START_DATE,
            end=end or config.END_DATE,
            auto_adjust=True,  # <- precios ajustados por splits y dividendos
            progress=False,
            threads=True,      # descarga en paralelo; sin esto tarda muchisimo
        )
        if downloaded.empty:
            raise RuntimeError("Yahoo Finance devolvio una tabla vacia.")

        prices = _extract_close(downloaded)
        prices.index.name = "date"
        return prices.sort_index()

    return _load_with_fallback(
        cache=cache or config.PRICES_CACHE,
        fallback=fallback or config.PRICES_FALLBACK,
        download=download,
        label="Yahoo Finance / precios",
        force_refresh=force_refresh,
    )


# ==========================================================================
# 3. Series macro (API de FRED)
# ==========================================================================
DEFAULT_FRED_SERIES = {"vix": "VIXCLS", "treasury_10y": "DGS10"}


def get_fred_series(
    series_ids: dict[str, str] | None = None,
    start: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Descarga series macro de FRED (Federal Reserve Bank of St. Louis).

    La llave se lee de ``FRED_API_KEY`` en el archivo ``.env`` (ver
    ``.env.example``); **nunca** se escribe en el codigo ni se sube al
    repositorio.

    Si no hay llave configurada, la funcion **no falla**: usa el respaldo
    incluido en el repositorio, y si tampoco lo hay, avisa y devuelve una tabla
    vacia. Quien revise el proyecto no necesita registrarse en FRED para poder
    ejecutar el notebook.

    Parameters
    ----------
    series_ids
        Mapeo ``{nombre_legible: id_de_la_serie_en_FRED}``. Por defecto trae el
        VIX (indice de volatilidad, el "termometro del miedo" del mercado) y la
        tasa del Tesoro a 10 anios.
    """
    if config.MACRO_CACHE.exists() and not force_refresh:
        return pd.read_parquet(config.MACRO_CACHE)

    if not config.FRED_API_KEY:
        if config.MACRO_FALLBACK.exists():
            warnings.warn(
                "[FRED] No hay FRED_API_KEY configurada. Se usa el respaldo "
                "incluido en el repositorio. Para descargar datos frescos, "
                "copia .env.example a .env y agrega tu llave gratuita de "
                "https://fredaccount.stlouisfed.org/apikeys",
                stacklevel=2,
            )
            return pd.read_parquet(config.MACRO_FALLBACK)

        warnings.warn(
            "[FRED] No hay FRED_API_KEY ni respaldo disponible. Se devuelve una "
            "tabla vacia; el resto del analisis continua sin datos macro.",
            stacklevel=2,
        )
        return pd.DataFrame()

    def download() -> pd.DataFrame:
        columns = {}
        for name, series_id in (series_ids or DEFAULT_FRED_SERIES).items():
            response = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": config.FRED_API_KEY,
                    "file_type": "json",
                    "observation_start": start or config.START_DATE,
                },
                timeout=30,
            )
            response.raise_for_status()
            observations = pd.DataFrame(response.json()["observations"])

            # FRED marca los datos faltantes con un punto ".", no con nulos.
            # errors="coerce" los convierte en NaN en lugar de reventar.
            values = pd.to_numeric(observations["value"], errors="coerce")
            values.index = pd.to_datetime(observations["date"])
            columns[name] = values

        macro = pd.DataFrame(columns)
        macro.index.name = "date"
        return macro

    return _load_with_fallback(
        cache=config.MACRO_CACHE,
        fallback=config.MACRO_FALLBACK,
        download=download,
        label="FRED / macro",
        force_refresh=force_refresh,
    )


# ==========================================================================
# 4. Universo ampliado: S&P Composite 1500 (Entregable 2)
# ==========================================================================
def _download_index_members(url: str, cap_tier: str) -> pd.DataFrame:
    """Descarga los constituyentes de uno de los tres indices S&P."""
    response = requests.get(url, headers=_BROWSER_UA, timeout=30)
    response.raise_for_status()

    # Las tres paginas de Wikipedia tienen la tabla de constituyentes como
    # primera tabla, pero se busca por columnas en lugar de asumir la posicion:
    # si Wikipedia reordena la pagina, esto sigue funcionando.
    table = None
    for candidate in pd.read_html(StringIO(response.text)):
        columns = [str(c) for c in candidate.columns]
        if any(c in ("Symbol", "Ticker") for c in columns) and "GICS Sector" in columns:
            table = candidate
            break

    if table is None:
        raise RuntimeError(
            f"No se encontro la tabla de constituyentes en {url}. "
            "Probablemente cambio la estructura de la pagina."
        )

    symbol_col = "Symbol" if "Symbol" in table.columns else "Ticker"
    return pd.DataFrame(
        {
            "ticker": table[symbol_col].map(to_yahoo_ticker),
            "company": table["Security"].astype("string"),
            "sector": table["GICS Sector"].astype("string"),
            "sub_industry": table["GICS Sub-Industry"].astype("string"),
            "cap_tier": cap_tier,
        }
    )


def _download_sp1500_constituents() -> pd.DataFrame:
    partes = [
        _download_index_members(url, cap_tier)
        for cap_tier, url in config.SP_INDEX_URLS.items()
    ]
    combinado = pd.concat(partes, ignore_index=True)

    # Un ticker no deberia aparecer en dos indices a la vez, pero si pasa se
    # conserva el de mayor capitalizacion: es el indice al que realmente
    # pertenece, y duplicarlo romperia el conteo de pares unicos mas adelante.
    orden = {"large": 0, "mid": 1, "small": 2}
    combinado["_orden"] = combinado["cap_tier"].map(orden)
    combinado = (
        combinado.sort_values(["ticker", "_orden"])
        .drop_duplicates(subset="ticker", keep="first")
        .drop(columns="_orden")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    combinado["cap_tier"] = combinado["cap_tier"].astype("string")
    return combinado


def get_sp1500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """Constituyentes del S&P Composite 1500 con su nivel de capitalizacion.

    Union de los tres indices: S&P 500 (large cap), S&P MidCap 400 y S&P
    SmallCap 600. Triplica el universo del Entregable 1 reutilizando la misma
    tecnica de scraping.

    La columna ``cap_tier`` (large / mid / small) es nueva y es la que habilita
    la pregunta del Entregable 2: depende menos del mercado una empresa chica
    que una grande?

    Returns
    -------
    DataFrame con: ticker, company, sector, sub_industry, cap_tier.
    """
    return _load_with_fallback(
        cache=config.SP1500_CONSTITUENTS_CACHE,
        fallback=config.SP1500_CONSTITUENTS_FALLBACK,
        download=_download_sp1500_constituents,
        label="Wikipedia / S&P 1500",
        force_refresh=force_refresh,
    )
