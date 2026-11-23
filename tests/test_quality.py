"""Pruebas del diagnostico de calidad.

Por que existen
---------------
Un diagnostico de calidad falla de dos formas, y las dos son silenciosas:

* **Deja pasar datos rotos.** Un split mal aplicado o un precio congelado no
  producen ningun error: producen una correlacion que parece razonable.
* **Borra datos reales.** Un filtro demasiado agresivo descarta justo los
  eventos que el analisis tiene que ver -una bancarrota, un desplome- y el
  sesgo que introduce es igual de grave que conservar datos falsos.

Por eso cada criterio se prueba por sus dos lados: que atrapa lo que debe, y
que no atrapa lo que no debe. Las fronteras de los umbrales se fijan de forma
explicita (``>`` contra ``>=``) porque cambiarlas mueve en silencio cuantos
tickers se excluyen.

Los paneles son sinteticos y chicos, construidos para que la respuesta correcta
se pueda calcular a mano. Al final, una clase fija las cifras publicadas sobre
los datos reales del repositorio, y otra verifica que la version en Spark de
``nulos_por_anio`` da lo mismo que la de pandas; esa se omite sola si
``pyspark`` no esta instalado, como pasa en integracion continua.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pathlib
import re

import pytest

from src import config, quality


def dias(n: int, inicio: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(inicio, periods=n)


def panel(columnas: dict[str, list[float]], inicio: str = "2021-01-04") -> pd.DataFrame:
    """Panel de precios con dias habiles consecutivos."""
    n = len(next(iter(columnas.values())))
    return pd.DataFrame(columnas, index=dias(n, inicio), dtype=float)


def log_rendimientos(precios: pd.DataFrame) -> pd.DataFrame:
    return np.log(precios / precios.shift(1)).iloc[1:]


# ==========================================================================
# structure_summary
# ==========================================================================
class TestStructureSummary:
    def test_cuenta_celdas_y_faltantes(self):
        precios = panel({"A": [1, 2, np.nan], "B": [1, 2, 3]})
        resumen = quality.structure_summary(precios)
        assert resumen["observaciones totales"] == 6
        assert resumen["valores faltantes"] == 1
        assert resumen["% faltantes"] == pytest.approx(16.67)

    def test_detecta_fechas_y_tickers_duplicados(self):
        """Una fecha duplicada cuenta dos veces el rendimiento de ese dia.

        No truena nada: simplemente pesa doble en la correlacion. El resumen es
        el unico lugar donde se ve, asi que tiene que contarlo.
        """
        idx = dias(2)
        precios = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.DatetimeIndex([idx[0], idx[1], idx[1]]),
            columns=["A", "A"],
        )
        resumen = quality.structure_summary(precios)
        assert resumen["fechas duplicadas"] == 1
        assert resumen["tickers duplicados"] == 1


# ==========================================================================
# per_ticker_report
# ==========================================================================
class TestPerTickerReport:
    @pytest.fixture
    def precios(self):
        return panel({
            "LARGO": [1, 2, 3, 4],
            "JUSTO": [np.nan, 2, 3, 4],
            "CORTO": [np.nan, np.nan, 3, 4],
        })

    def test_la_frontera_de_historia_es_inclusiva(self, precios, monkeypatch):
        """``n_obs >= MIN_OBSERVATIONS``, no ``>``.

        Con el umbral real (2,900) mover la frontera un dia cambia que empresas
        entran al panel balanceado, y con ellas las cifras publicadas.
        """
        monkeypatch.setattr(config, "MIN_OBSERVATIONS", 3)
        reporte = quality.per_ticker_report(precios)
        assert reporte.loc["LARGO", "suficiente_historia"]
        assert reporte.loc["JUSTO", "suficiente_historia"]
        assert not reporte.loc["CORTO", "suficiente_historia"]

    def test_reporta_desde_cuando_cotiza_cada_ticker(self, precios):
        """La primera fecha es lo que distingue una empresa joven de un hueco."""
        reporte = quality.per_ticker_report(precios)
        assert reporte.loc["CORTO", "first_date"] == precios.index[2]
        assert reporte.loc["CORTO", "last_date"] == precios.index[3]
        assert reporte.loc["CORTO", "pct_missing"] == pytest.approx(50.0)

    def test_ordena_del_mas_corto_al_mas_largo(self, precios):
        reporte = quality.per_ticker_report(precios)
        assert list(reporte.index) == ["CORTO", "JUSTO", "LARGO"]


# ==========================================================================
# find_extreme_returns
# ==========================================================================
class TestFindExtremeReturns:
    @pytest.fixture
    def rendimientos(self):
        return pd.DataFrame(
            {"X": [0.01, 0.60, -0.70, 0.50], "Y": [np.nan, -0.51, 0.0, 0.02]},
            index=dias(4),
        )

    def test_el_umbral_es_estricto(self, rendimientos):
        """Un rendimiento de exactamente 0.50 no cuenta como extremo.

        La frontera se fija porque ``find_extreme_returns`` y
        ``flag_suspicious_tickers`` deben usar la misma: si una usara ``>=``,
        la tabla que se revisa a mano y el filtro automatico no coincidirian.
        """
        extremos = quality.find_extreme_returns(rendimientos)
        assert 0.50 not in extremos["log_return"].tolist()
        assert len(extremos) == 3

    def test_atrapa_subidas_y_bajadas(self, rendimientos):
        extremos = quality.find_extreme_returns(rendimientos)
        assert (extremos["log_return"] > 0).any()
        assert (extremos["log_return"] < 0).any()

    def test_ordena_por_magnitud_sin_importar_el_signo(self, rendimientos):
        """El caso mas grave tiene que salir primero, sea caida o subida."""
        extremos = quality.find_extreme_returns(rendimientos)
        assert extremos["log_return"].tolist() == [-0.70, 0.60, -0.51]

    def test_devuelve_las_columnas_documentadas(self, rendimientos):
        extremos = quality.find_extreme_returns(rendimientos)
        assert list(extremos.columns) == ["date", "ticker", "log_return"]
        fila = extremos.iloc[0]
        assert fila["ticker"] == "X"
        assert fila["date"] == rendimientos.index[2]

    def test_no_devuelve_el_panel_completo_con_nulos(self):
        """Fija el ``.dropna()`` que exige pandas 3.

        Desde pandas 3.0 ``.stack()`` conserva los nulos. Sin el ``dropna`` la
        mascara deja NaN en todas las celdas que no son extremas, y la funcion
        devolveria el panel entero en lugar de un caso. Es un fallo que solo
        aparece al actualizar pandas, asi que nadie lo veria venir.
        """
        rendimientos = pd.DataFrame(
            np.full((50, 20), 0.001), index=dias(50),
            columns=[f"T{i}" for i in range(20)],
        )
        rendimientos.iloc[::3, ::2] = np.nan
        rendimientos.iloc[10, 5] = -0.9
        extremos = quality.find_extreme_returns(rendimientos)
        assert len(extremos) == 1
        assert extremos.iloc[0]["ticker"] == "T5"

    def test_sin_extremos_devuelve_tabla_vacia(self):
        rendimientos = pd.DataFrame({"A": [0.01, -0.02]}, index=dias(2))
        extremos = quality.find_extreme_returns(rendimientos)
        assert extremos.empty
        assert list(extremos.columns) == ["date", "ticker", "log_return"]

    def test_respeta_un_umbral_personalizado(self, rendimientos):
        extremos = quality.find_extreme_returns(rendimientos, threshold=0.65)
        assert extremos["log_return"].tolist() == [-0.70]


# ==========================================================================
# flag_suspicious_tickers
# ==========================================================================
def _split_mal_aplicado() -> list[float]:
    """Tipo MNST: el precio cae a la mitad y regresa, tres veces."""
    return [100, 100, 50, 100, 100, 50, 100, 100, 50, 100] + [100] * 10


def _desplome_real() -> list[float]:
    """Tipo PCG: una caida de mas de la mitad que no se recupera."""
    return [100] * 10 + [45] * 10


def _volatil_pero_real() -> list[float]:
    """Tipo GME: varias subidas enormes que NO regresan.

    Es el caso que el criterio por frecuencia clasificaba mal. GameStop subio
    mas de 1,000% en 2021 en episodios separados, y cada nivel se sostuvo. Son
    datos correctos por mas extremos que parezcan.
    """
    return [10] * 5 + [25] * 5 + [60] * 5 + [150] * 5


class TestFlagSuspiciousTickers:
    @pytest.fixture
    def rendimientos(self):
        return log_rendimientos(panel({
            "MNST": _split_mal_aplicado(),
            "PCG": _desplome_real(),
            "GME": _volatil_pero_real(),
            "KO": [100] * 20,
        }))

    def test_marca_un_split_mal_aplicado(self, rendimientos):
        """El caso que motivo el criterio: MNST en el Entregable 1.

        Cada ida y vuelta son dos movimientos de +/-69%, asi que tres
        oscilaciones dan seis eventos. Lo que lo delata no es cuantos son sino
        que el precio **regresa** cada vez.
        """
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.loc["MNST", "n_eventos_extremos"] == 6
        assert reporte.loc["MNST", "n_reversiones"] >= 3
        assert reporte.loc["MNST", "sospechoso"]

    def test_conserva_una_accion_muy_volatil_pero_correcta(self, rendimientos):
        """El caso que el criterio por frecuencia clasificaba mal.

        Tres subidas de mas del 100% que nunca se deshacen. Contando eventos,
        son tres y se excluiria; mirando reversiones, son cero y se conserva.
        Detectado por Miguel Herrera al escribir estas pruebas: el criterio de
        reversion vivia solo en el notebook de Spark, y la version de pandas
        habia quedado con el criterio viejo.
        """
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.loc["GME", "n_eventos_extremos"] >= 3
        assert reporte.loc["GME", "n_reversiones"] == 0
        assert not reporte.loc["GME", "sospechoso"]

    def test_conserva_un_desplome_real(self, rendimientos):
        """Un solo desplome es un evento, no un error de la fuente.

        Es el lado que importa tanto como el otro: PCG en 2019 fue la bancarrota
        de PG&E. Borrarlo sesgaria las correlaciones de Utilities hacia arriba,
        porque quitaria justo el dia en que una electrica se separo de las demas.
        """
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.loc["PCG", "n_eventos_extremos"] == 1
        assert not reporte.loc["PCG", "sospechoso"]

    def test_tolera_una_reversion_suelta(self):
        """``n_reversiones > MAX_REVERSIONES``: con exactamente una no se marca.

        Una reversion aislada puede ser un dato malo suelto y no un split mal
        aplicado. Si la comparacion se volviera ``>=``, un ticker con un solo
        tropiezo quedaria fuera sin que ninguna otra prueba lo note.
        """
        rendimientos = log_rendimientos(panel({
            "UNA": [100] * 5 + [45, 100] + [100] * 8,
        }))
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.loc["UNA", "n_reversiones"] == config.MAX_REVERSIONES
        assert not reporte.loc["UNA", "sospechoso"]

    def test_los_eventos_se_reparten_entre_reales_y_revertidos(self, rendimientos):
        """Identidad de la tabla: cada evento extremo es una cosa o la otra."""
        reporte = quality.flag_suspicious_tickers(rendimientos)
        suma = reporte["n_reversiones"] + reporte["n_reales"]
        assert (suma == reporte["n_eventos_extremos"]).all()

    def test_omite_los_tickers_sin_eventos(self, rendimientos):
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert "KO" not in reporte.index

    def test_ordena_del_mas_sospechoso_al_menos(self, rendimientos):
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.index[0] == "MNST"

    def test_respeta_los_parametros(self, rendimientos):
        reporte = quality.flag_suspicious_tickers(rendimientos, max_reversiones=10)
        assert not reporte["sospechoso"].any()

    def test_una_reversion_mas_lenta_tambien_cuenta(self):
        """El regreso no siempre ocurre al dia siguiente.

        Por eso se miran tres sesiones hacia adelante y no solo una. Aqui el
        precio tarda dos dias en volver, y debe detectarse igual.
        """
        rendimientos = log_rendimientos(panel({
            "LENTO": [100] * 4 + [45, 60, 100] + [100] * 4
                     + [45, 60, 100] + [100] * 4,
        }))
        reporte = quality.flag_suspicious_tickers(rendimientos)
        assert reporte.loc["LENTO", "n_reversiones"] >= 2
        assert reporte.loc["LENTO", "sospechoso"]


# ==========================================================================
# calendar_gaps
# ==========================================================================
class TestCalendarGaps:
    def test_un_fin_de_semana_largo_no_es_un_hueco(self):
        """Viernes a martes son 4 dias: un feriado, no un dato perdido."""
        fechas = pd.to_datetime(["2021-01-14", "2021-01-15", "2021-01-19"])
        precios = pd.DataFrame({"A": [1, 2, 3]}, index=fechas)
        assert quality.calendar_gaps(precios).empty

    def test_detecta_un_hueco_largo(self):
        fechas = pd.to_datetime(["2021-01-15", "2021-01-19", "2021-02-01"])
        precios = pd.DataFrame({"A": [1, 2, 3]}, index=fechas)
        huecos = quality.calendar_gaps(precios)
        assert len(huecos) == 1
        assert huecos.iloc[0]["fecha"] == pd.Timestamp("2021-02-01")
        assert huecos.iloc[0]["dias_sin_datos"] == 13


# ==========================================================================
# nulos_por_anio
# ==========================================================================
@pytest.fixture
def panel_con_los_tres_huecos() -> pd.DataFrame:
    """Tres dias de 2021 y tres de 2022, con un ticker por tipo de hueco.

        NUEVO   entra a cotizar el tercer dia   -> 2 previos al listado (2021)
        BAJA    deja de cotizar en 2022         -> 3 posteriores a la baja
        HUECO   un dia vacio en cada anio       -> 1 interno por anio
    """
    fechas = pd.to_datetime([
        "2021-12-29", "2021-12-30", "2021-12-31",
        "2022-01-03", "2022-01-04", "2022-01-05",
    ])
    return pd.DataFrame({
        "COMPLETO": [1, 2, 3, 4, 5, 6],
        "NUEVO": [np.nan, np.nan, 10, 10, 11, 12],
        "BAJA": [5, 5, 6, np.nan, np.nan, np.nan],
        "HUECO": [1, np.nan, 2, 3, np.nan, 4],
    }, index=fechas, dtype=float)


class TestNulosPorAnio:
    def test_clasifica_cada_hueco_por_su_causa(self, panel_con_los_tres_huecos):
        """El punto de la funcion: separar el hueco normal del preocupante.

        Una empresa que aun no cotiza o que ya salio deja celdas vacias que no
        son datos perdidos. Solo los internos lo son. Si la clasificacion se
        mezclara, el diagnostico volveria a decir "hay muchos huecos" sin poder
        decir si importan.
        """
        anual = quality.nulos_por_anio(panel_con_los_tres_huecos)
        assert anual.loc[2021, "previos_al_listado"] == 2
        assert anual.loc[2021, "posteriores_a_la_baja"] == 0
        assert anual.loc[2021, "huecos_internos"] == 1
        assert anual.loc[2022, "previos_al_listado"] == 0
        assert anual.loc[2022, "posteriores_a_la_baja"] == 3
        assert anual.loc[2022, "huecos_internos"] == 1

    def test_el_desglose_cuadra_con_el_total(self, panel_con_los_tres_huecos):
        anual = quality.nulos_por_anio(panel_con_los_tres_huecos)
        partes = anual[["previos_al_listado", "posteriores_a_la_baja",
                        "huecos_internos"]].sum(axis=1)
        pd.testing.assert_series_equal(partes, anual["n_faltantes"],
                                       check_names=False)
        assert anual["n_faltantes"].sum() == panel_con_los_tres_huecos.isna().sum().sum()

    def test_normaliza_por_los_dias_de_cada_anio(self, panel_con_los_tres_huecos):
        """``prom_por_dia`` existe porque los anios no tienen los mismos dias.

        El ultimo anio del periodo siempre esta incompleto. Comparar conteos
        crudos haria parecer que los huecos desaparecen solo por tener menos
        dias.
        """
        anual = quality.nulos_por_anio(panel_con_los_tres_huecos)
        assert anual.loc[2022, "prom_por_dia"] == pytest.approx(1.3)
        assert anual.loc[2022, "pct_faltantes"] == pytest.approx(33.33)


# ==========================================================================
# calidad_por_sector
# ==========================================================================
class TestCalidadPorSector:
    @pytest.fixture
    def universo(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_OBSERVATIONS", 3)
        completo, corto = [1, 2, 3, 4], [np.nan, np.nan, 3, 4]
        grande = {f"G{i}": (corto if i < 2 else completo) for i in range(10)}
        chico = {f"C{i}": (corto if i < 1 else completo) for i in range(3)}
        precios = panel({**grande, **chico})
        constituyentes = pd.DataFrame({
            "ticker": list(grande) + list(chico),
            "sector": ["Grande"] * 10 + ["Chico"] * 3,
        })
        return precios, constituyentes

    def test_ordena_por_proporcion_y_no_por_conteo(self, universo):
        """El hallazgo del Entregable 1 vive en el porcentaje.

        Grande pierde 2 de 10 (20%) y Chico 1 de 3 (33%). En conteo crudo
        Grande parece el mas castigado; en proporcion lo es Chico. Es el caso
        de Communication Services (5 de 24) frente a Industrials (11 de 83).
        Ordenar por conteo esconderia el sector peor medido.
        """
        precios, constituyentes = universo
        resumen = quality.calidad_por_sector(precios, constituyentes)
        assert resumen.index[0] == "Chico"
        assert resumen.loc["Grande", "descartados"] > resumen.loc["Chico", "descartados"]
        assert resumen.loc["Chico", "pct_descartados"] == pytest.approx(33.3)

    def test_un_ticker_sin_sector_no_desaparece(self, universo):
        """Sin esto, el ``groupby`` lo tiraria en silencio.

        El total por sector dejaria de sumar el universo, y nadie sabria que
        falto una empresa.
        """
        precios, constituyentes = universo
        precios = precios.assign(HUERFANO=[1, 2, 3, 4])
        resumen = quality.calidad_por_sector(precios, constituyentes)
        assert "SIN CLASIFICAR" in resumen.index
        assert resumen["tickers"].sum() == precios.shape[1]


# ==========================================================================
# detectar_precios_estancados
# ==========================================================================
class TestDetectarPreciosEstancados:
    def test_reporta_la_racha_con_sus_fechas(self):
        precios = panel({"A": [10, 11, 12, 12, 12, 12, 13]})
        estancados = quality.detectar_precios_estancados(precios, dias_minimos=3)
        fila = estancados.iloc[0]
        assert fila["dias_sin_moverse"] == 3
        assert fila["fecha_inicio"] == precios.index[2].date()
        assert fila["fecha_fin"] == precios.index[5].date()
        assert fila["precio"] == 12

    def test_la_frontera_de_dias_es_inclusiva(self):
        """Con el default, 10 dias sin moverse se reportan y 9 no.

        Diez es el criterio que usa la capa Silver para excluir por precio
        estancado. Si la frontera se moviera, cambiaria que tickers se excluyen
        sin que ningun umbral de configuracion lo diga.
        """
        precios = panel({
            "NUEVE": [1] + [2] * 10 + [3],
            "DIEZ": [1] + [2] * 11,
        })
        estancados = quality.detectar_precios_estancados(precios)
        assert estancados["ticker"].tolist() == ["DIEZ"]
        assert estancados.iloc[0]["dias_sin_moverse"] == 10

    def test_un_dia_sin_dato_corta_la_racha(self):
        """Un hueco no es evidencia de que el precio se congelo.

        Si la funcion rellenara hacia adelante, el dia vacio contaria como "sin
        cambio" y uniria dos rachas cortas en una larga: exactamente el error
        que el proyecto evita al no usar ``ffill``.
        """
        precios = panel({"A": [5, 5, 5, np.nan, 5, 5, 5, 9]})
        assert quality.detectar_precios_estancados(precios, dias_minimos=3).empty
        estancados = quality.detectar_precios_estancados(precios, dias_minimos=2)
        assert estancados.iloc[0]["dias_sin_moverse"] == 2

    def test_reporta_la_racha_mas_larga_y_no_la_suma(self):
        precios = panel({"A": [1, 1, 1, 1, 2, 2, 2, 2]})
        estancados = quality.detectar_precios_estancados(precios, dias_minimos=2)
        assert estancados.iloc[0]["dias_sin_moverse"] == 3

    def test_un_precio_que_se_mueve_no_se_reporta(self):
        rng = np.random.default_rng(0)
        precios = panel({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))})
        assert quality.detectar_precios_estancados(precios).empty

    def test_sin_casos_devuelve_tabla_vacia_con_columnas(self):
        """Con un umbral alto y ningun caso, no debe tronar.

        Antes de este arreglo, ``sort_values`` sobre un DataFrame sin columnas
        lanzaba KeyError. Pasa justo cuando los datos estan limpios, que es
        cuando nadie espera un error.
        """
        precios = panel({"A": [1, 1, 2, 3]})
        estancados = quality.detectar_precios_estancados(precios, dias_minimos=500)
        assert estancados.empty
        assert list(estancados.columns) == [
            "ticker", "dias_sin_moverse", "fecha_inicio", "fecha_fin", "precio",
        ]

    def test_ordena_de_la_racha_mas_larga_a_la_mas_corta(self):
        precios = panel({
            "CORTA": [1, 2, 2, 2, 3, 4, 5, 6],
            "LARGA": [1, 2, 2, 2, 2, 2, 2, 3],
        })
        estancados = quality.detectar_precios_estancados(precios, dias_minimos=2)
        assert estancados["ticker"].tolist() == ["LARGA", "CORTA"]


# ==========================================================================
# nulos_por_anio_spark
# ==========================================================================
@pytest.fixture(scope="module")
def spark():
    """Sesion local de Spark; la prueba se omite si pyspark no esta instalado."""
    pytest.importorskip("pyspark")
    import os
    import sys

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    sesion = (
        SparkSession.builder.master("local[1]")
        .appName("test_quality")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


class TestNulosPorAnioSpark:
    """La version Spark tiene que contar lo mismo que la de pandas.

    En formato largo un faltante no es un nulo, es una fila que no existe. La
    forma intuitiva de contarlos -tickers de cada anio por dias de ese anio-
    pierde a las empresas que aun no cotizaban, porque no tienen ninguna fila
    en ese anio. Sobre el panel del Entregable 1 ese error encontraba 5,156
    faltantes en lugar de 54,479. Comparar contra pandas es la unica forma de
    atrapar ese tipo de error, porque el resultado equivocado tambien parece
    razonable.
    """

    @staticmethod
    def _a_formato_largo(spark, ancho: pd.DataFrame, nombre: str) -> None:
        """Igual que la ingesta de Bronze: ``stack().dropna()``."""
        largo = ancho.stack().dropna().rename("close").reset_index()
        largo.columns = ["date", "ticker", "close"]
        largo["date"] = largo["date"].dt.date
        spark.createDataFrame(largo).createOrReplaceTempView(nombre)

    def test_coincide_con_la_version_de_pandas(self, spark, panel_con_los_tres_huecos):
        self._a_formato_largo(spark, panel_con_los_tres_huecos, "precios_largos")
        obtenido = (
            quality.nulos_por_anio_spark(spark, "precios_largos")
            .toPandas().set_index("anio")
        )
        esperado = quality.nulos_por_anio(panel_con_los_tres_huecos)
        for columna in esperado.columns:
            assert obtenido[columna].astype(float).tolist() == pytest.approx(
                esperado[columna].astype(float).tolist()
            ), columna

    def test_cuenta_a_las_empresas_que_aun_no_cotizaban(self, spark,
                                                       panel_con_los_tres_huecos):
        """El error especifico que la funcion corrige, fijado aparte.

        NUEVO no tiene filas en sus primeros dos dias. Si la funcion solo
        esperara a los tickers presentes en cada fecha, esos dos huecos no
        existirian.
        """
        self._a_formato_largo(spark, panel_con_los_tres_huecos, "precios_largos")
        obtenido = (
            quality.nulos_por_anio_spark(spark, "precios_largos")
            .toPandas().set_index("anio")
        )
        assert obtenido.loc[2021, "previos_al_listado"] == 2

    def test_las_filas_duplicadas_no_cuentan_doble(self, spark,
                                                  panel_con_los_tres_huecos):
        """Un lote escrito dos veces en Bronze no debe cambiar el conteo."""
        self._a_formato_largo(spark, panel_con_los_tres_huecos, "precios_largos")
        duplicada = spark.table("precios_largos").unionByName(
            spark.table("precios_largos")
        )
        duplicada.createOrReplaceTempView("precios_duplicados")
        normal = quality.nulos_por_anio_spark(spark, "precios_largos").toPandas()
        doble = quality.nulos_por_anio_spark(spark, "precios_duplicados").toPandas()
        pd.testing.assert_frame_equal(normal, doble)


# ==========================================================================
# Cifras publicadas sobre los datos reales
# ==========================================================================
HAY_RESPALDO = config.PRICES_FALLBACK.exists() and config.CONSTITUENTS_FALLBACK.exists()
sin_respaldo = pytest.mark.skipif(
    not HAY_RESPALDO, reason="requiere los parquet de data/processed/"
)


@pytest.fixture(scope="module")
def precios_reales() -> pd.DataFrame:
    return pd.read_parquet(config.PRICES_FALLBACK)


@sin_respaldo
class TestCifrasPublicadas:
    """Las cifras del diagnostico que citan el notebook y la documentacion.

    Se leen directo del respaldo versionado y no de ``data.get_prices``, para
    que el resultado no dependa de si hay cache local ni de la red.
    """

    def test_los_precios_estancados_conocidos(self, precios_reales):
        """AMCR, SW y FERG, documentados en el Entregable 1 y en el 2."""
        estancados = quality.detectar_precios_estancados(precios_reales)
        assert dict(zip(estancados["ticker"], estancados["dias_sin_moverse"])) == {
            "AMCR": 66, "SW": 34, "FERG": 19,
        }

    def test_los_faltantes_son_empresas_que_aun_no_cotizaban(self, precios_reales):
        """De 54,479 celdas vacias, solo 2 son huecos internos."""
        anual = quality.nulos_por_anio(precios_reales)
        assert anual["n_faltantes"].sum() == 54_479
        assert anual["huecos_internos"].sum() == 2
        assert anual.loc[2015, "prom_por_dia"] == pytest.approx(41.5)
        assert anual.loc[2026, "prom_por_dia"] == pytest.approx(1.3)
        assert anual["prom_por_dia"].is_monotonic_decreasing

    def test_communication_services_es_el_sector_peor_medido(self, precios_reales):
        constituyentes = pd.read_parquet(config.CONSTITUENTS_FALLBACK)
        resumen = quality.calidad_por_sector(precios_reales, constituyentes)
        assert resumen.index[0] == "Communication Services"
        assert resumen.loc["Communication Services", "descartados"] == 5
        assert resumen.loc["Communication Services", "tickers"] == 24


# ==========================================================================
# Coherencia entre las dos implementaciones del criterio
# ==========================================================================
class TestLosDosCriteriosNoSePuedenSeparar:
    """El mismo criterio vive en dos sitios y tiene que decir lo mismo.

    ``src/quality.py`` lo aplica sobre pandas para el Entregable 1 y
    ``databricks/02_silver_limpieza.py`` sobre Spark para el 2. Ya se
    separaron una vez: el notebook adopto la deteccion por reversion y la
    version de pandas se quedo con el conteo por frecuencia, de modo que una
    accion muy volatil pero correcta quedaba marcada como dato corrupto segun
    por donde se mirara. Lo encontro Miguel Herrera escribiendo las pruebas de
    este modulo.

    Estas pruebas leen el notebook como texto y comparan sus umbrales contra
    los de ``config``. No pueden ejecutar Spark, pero si pueden impedir que los
    numeros vuelvan a divergir en silencio.
    """

    NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "databricks" / "02_silver_limpieza.py"

    def _constante(self, nombre: str) -> float:
        texto = self.NOTEBOOK.read_text(encoding="utf-8")
        hallado = re.search(rf"^{nombre}\s*=\s*([0-9.]+)", texto, re.M)
        assert hallado, f"{nombre} no esta definida en {self.NOTEBOOK.name}"
        return float(hallado.group(1))

    def test_el_notebook_existe(self):
        assert self.NOTEBOOK.is_file()

    def test_mismo_umbral_de_evento_extremo(self):
        assert self._constante("UMBRAL_EXTREMO") == config.EXTREME_RETURN_THRESHOLD

    def test_misma_tolerancia_de_reversion(self):
        assert self._constante("TOLERANCIA_REVERSION") == config.TOLERANCIA_REVERSION

    def test_mismos_dias_para_revertir(self):
        assert self._constante("DIAS_PARA_REVERTIR") == config.DIAS_REVERSION

    def test_mismo_maximo_de_reversiones(self):
        assert self._constante("MAX_REVERSIONES") == config.MAX_REVERSIONES
