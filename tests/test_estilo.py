"""Protege la hoja de estilos del tablero de un error concreto y repetido.

Por que existe este archivo
---------------------------
Esconder la cabecera de Streamlit con ``display: none`` parece inofensivo: ahi
viven el boton de Deploy y el menu de hamburguesa, que no pintan nada en un
tablero de entrega. Pero cuando la barra lateral se pliega, Streamlit inserta
**dentro de esa misma cabecera** el unico boton que la reabre. Al esconderla
entera, quien plegaba la barra se quedaba sin menu, sin navegacion y sin forma
de recuperarla; la unica salida era recargar la pagina.

Lo reporto Franco usandolo. La correccion es neutralizar la cabecera -fondo
transparente y ``pointer-events: none``- y esconder uno por uno los controles
que sobran, dejando vivo el boton de reabrir.

Estas pruebas leen la hoja de estilos que el tablero inyecta de verdad, no una
copia. Si alguien vuelve a escribir la regla de siempre, fallan.
"""

from __future__ import annotations

import re

import pytest

estilo = pytest.importorskip("dashboard.estilo", reason="requiere streamlit")


@pytest.fixture(scope="module")
def hoja(request) -> str:
    """La hoja de estilos real, capturada al vuelo.

    Se sustituye ``estilo.html`` en vez de leer el archivo como texto: asi se
    prueba lo que el tablero inyecta y no lo que esta escrito antes de
    interpolar la paleta.
    """
    capturado: list[str] = []
    original = estilo.html
    estilo.html = capturado.append
    try:
        estilo.aplicar_estilo()
    finally:
        estilo.html = original
    assert capturado, "aplicar_estilo no inyecto nada"
    return capturado[0]


class TestLaCabeceraSigueAlcanzable:
    def test_la_cabecera_no_se_esconde_entera(self, hoja):
        """``display: none`` sobre la cabecera se lleva el boton de reabrir."""
        regla = re.search(
            r'header\[data-testid="stHeader"\]\s*\{([^}]*)\}', hoja
        )
        assert regla, "no se encontro la regla de la cabecera"
        cuerpo = regla.group(1)
        assert "display" not in cuerpo or "none" not in cuerpo, (
            "La cabecera no se puede esconder con display:none: ahi vive el "
            "boton que reabre la barra lateral. Neutralizala en su lugar."
        )

    def test_la_cabecera_no_intercepta_clics(self, hoja):
        """Queda visible pero transparente, asi que no debe comerse los clics."""
        regla = re.search(
            r'header\[data-testid="stHeader"\]\s*\{([^}]*)\}', hoja
        )
        assert "pointer-events" in regla.group(1)
        assert "none" in regla.group(1)

    def test_el_boton_de_reabrir_recupera_el_clic(self, hoja):
        """La cabecera los desactiva en bloque; este tiene que reactivarlos."""
        regla = re.search(
            r'button\[data-testid="stExpandSidebarButton"\]\s*\{([^}]*)\}', hoja
        )
        assert regla, "no se encontro la regla del boton de reabrir"
        cuerpo = regla.group(1)
        assert "pointer-events" in cuerpo and "auto" in cuerpo, (
            "Sin pointer-events:auto el boton se ve pero no se puede pulsar."
        )

    def test_el_selector_apunta_al_boton_y_no_a_un_contenedor(self, hoja):
        """``stExpandSidebarButton`` **es** el <button>, no lo envuelve.

        La primera version del arreglo escribio
        ``[data-testid="stExpandSidebarButton"] button``, que no encaja con
        nada: el boton quedaba sin estilo y casi fuera de la pantalla.
        """
        assert '[data-testid="stExpandSidebarButton"] button' not in hoja

    @pytest.mark.parametrize("control", [
        "stToolbarActions",
        "stAppDeployButton",
        "stMainMenu",
    ])
    def test_los_controles_de_streamlit_si_se_esconden(self, hoja, control):
        """Lo que sobra se esconde uno por uno, no de un plumazo."""
        assert f'[data-testid="{control}"]' in hoja
