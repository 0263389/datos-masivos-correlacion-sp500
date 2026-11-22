"""Capa de presentacion: identidad visual y componentes del terminal.

El modelo a seguir
------------------
Una terminal de riesgo institucional -Aladdin de BlackRock, PORT de Bloomberg,
Barra de MSCI- no se parece a un reporte ni a una pagina de marketing. Tiene
cuatro rasgos que aqui se imitan a proposito:

* **Densidad.** La informacion se lee en tablas, no en tarjetas grandes. Quien
  la usa quiere comparar cien filas, no admirar cinco numeros.
* **Cifras tabulares.** Todo numero va monoespaciado, alineado a la derecha y
  con su unidad en el encabezado. Asi las columnas se comparan de un vistazo.
* **Color con significado.** Rojo y verde indican direccion del riesgo y nada
  mas. Si se usaran para decorar, dejarian de informar.
* **Procedencia.** Cada cifra dice de donde sale: universo, periodo, metodo.

Terminologia
------------
Los terminos tecnicos se conservan, porque son los correctos y son los que el
usuario va a encontrar en cualquier otro sistema. Lo que no se acepta es que
aparezcan sin explicacion: el componente ``termino`` marca cada uno y le pega
su definicion en lenguaje normal, y ``detalle`` guarda la formula y los
supuestos para quien los quiera revisar.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ==========================================================================
# 1. Paleta
# ==========================================================================
# Chrome oscuro (barra lateral y cabecera) sobre lienzo claro: es lo que separa
# visualmente una aplicacion de un documento.
NOCHE = "#101823"
NOCHE_2 = "#1a2533"
NOCHE_B = "#2a3747"
NIEBLA = "#8fa0b4"
HIELO = "#dde5ef"

TINTA = "#0b121b"
SUAVE = "#58677a"
TENUE = "#93a1b2"
LINEA = "#e3e8ef"
LINEA_F = "#cfd7e2"
FONDO = "#f6f8fa"
BLANCO = "#ffffff"

MARCA = "#1b5e8f"
MARCA_CLARA = "#eaf2f9"
ALTO = "#ab2f28"
ALTO_CLARO = "#fbeded"
BAJO = "#126b47"
BAJO_CLARO = "#e9f4ef"
AVISO = "#8f6410"
AVISO_CLARO = "#fdf5e6"

TONOS = {
    "neutro": (TINTA, LINEA_F, BLANCO),
    "marca": (MARCA, MARCA, MARCA_CLARA),
    "alto": (ALTO, ALTO, ALTO_CLARO),
    "bajo": (BAJO, BAJO, BAJO_CLARO),
    "aviso": (AVISO, AVISO, AVISO_CLARO),
}


def html(html: str) -> None:
    """Inyecta HTML quitandole los renglones vacios.

    Streamlit pasa el HTML por un procesador de Markdown antes de insertarlo, y
    eso impone dos restricciones sobre el contenido:

    * Un **renglon vacio** termina el bloque de HTML. Lo que venga despues se
      convierte en un parrafo de texto normal, asi que media hoja de estilos
      terminaria impresa en pantalla. Por eso se colapsan aqui.
    * Una linea con **signos de igual o guiones repetidos** se interpreta como
      el subrayado de un encabezado y corta el bloque igual. Por eso los
      comentarios del CSS de abajo no llevan separadores decorativos.

    No es cosmetico. Streamlit pasa el HTML por un procesador de Markdown, y
    Markdown considera que un renglon vacio **termina** un bloque de HTML: todo
    lo que viene despues se vuelve un parrafo de texto normal. Con una hoja de
    estilos eso significa que la mitad del CSS se imprime en pantalla en vez de
    aplicarse. Colapsar los renglones vacios deja un solo bloque continuo.
    """
    st.html("\n".join(ln for ln in html.split("\n") if ln.strip()))


def aplicar_estilo() -> None:
    """Inyecta la hoja de estilos. Se llama una sola vez, al arrancar."""
    html(f"""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
  :root {{
    --noche:{NOCHE}; --noche2:{NOCHE_2}; --nocheb:{NOCHE_B};
    --niebla:{NIEBLA}; --hielo:{HIELO};
    --tinta:{TINTA}; --suave:{SUAVE}; --tenue:{TENUE};
    --linea:{LINEA}; --lineaf:{LINEA_F}; --fondo:{FONDO}; --blanco:{BLANCO};
    --marca:{MARCA}; --marcac:{MARCA_CLARA};
    --alto:{ALTO}; --altoc:{ALTO_CLARO};
    --bajo:{BAJO}; --bajoc:{BAJO_CLARO};
    --aviso:{AVISO}; --avisoc:{AVISO_CLARO};
    --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  }}

  html, body, [class*="st-"], .stMarkdown, p, li, div, label, span {{
    font-family: var(--sans);
  }}
  /* Los iconos de Streamlit son una fuente de simbolos: si se les impone la
     tipografia de texto, el nombre del icono se imprime como palabra suelta. */
  [data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
  span[class*="material"], [data-testid*="Icon"] {{
    font-family: "Material Symbols Rounded", "Material Icons" !important;
  }}
  .stApp {{ background: var(--fondo); color: var(--tinta); }}
  .block-container {{
    padding: 1.1rem 2.2rem 4rem; max-width: 1340px;
    font-size: 13.5px; line-height: 1.55;
  }}
  header[data-testid="stHeader"] {{ display: none; }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1,h2,h3,h4 {{ font-family: var(--sans); letter-spacing: -.012em; color: var(--tinta); }}

  /* barra lateral oscura */
  section[data-testid="stSidebar"] {{ background: var(--noche); border: none; width: 252px !important; }}
  section[data-testid="stSidebar"] .block-container,
  section[data-testid="stSidebar"] > div {{ padding-top: 0; }}
  section[data-testid="stSidebar"] * {{ color: var(--hielo); }}
  section[data-testid="stSidebar"] label {{
    color: var(--niebla) !important; font-size: 10.5px !important;
    font-weight: 500 !important; letter-spacing: .09em !important;
    text-transform: uppercase !important;
  }}
  .logo {{
    padding: 20px 4px 16px; border-bottom: 1px solid var(--nocheb); margin-bottom: 16px;
  }}
  .logo-n {{
    font-size: 15px; font-weight: 700; letter-spacing: -.01em;
    color: #fff; line-height: 1.2;
  }}
  .logo-s {{
    font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--niebla); margin-top: 5px;
  }}
  .lat-tit {{
    font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--niebla);
    margin: 20px 0 8px; padding-top: 14px; border-top: 1px solid var(--nocheb);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 1px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 7px 10px; border-radius: 3px; margin: 0;
    font-size: 12.5px !important; font-weight: 500 !important;
    letter-spacing: 0 !important; text-transform: none !important;
    color: var(--hielo) !important; transition: background .12s;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: var(--noche2); }}
  /* El control nativo se oculta a la vista pero sigue existiendo para el
     teclado y los lectores de pantalla. display:none lo sacaria del arbol de
     accesibilidad y dejaria la navegacion inservible sin raton. */
  section[data-testid="stSidebar"] div[role="radiogroup"] input {{
    position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
    border: 0; overflow: hidden; white-space: nowrap; clip-path: inset(50%);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:focus-within {{
    outline: 2px solid var(--hielo); outline-offset: 1px;
  }}
  .lat-pie {{
    font-size: 10.5px; color: var(--niebla); line-height: 1.6;
    border-top: 1px solid var(--nocheb); margin-top: 22px; padding-top: 14px;
  }}
  .lat-pie b {{ color: var(--hielo); font-weight: 600; }}

  /* cabecera de la vista */
  .cab {{
    background: var(--blanco); border: 1px solid var(--linea);
    border-radius: 4px; padding: 16px 20px 15px; margin-bottom: 14px;
  }}
  .cab-via {{
    font-family: var(--mono); font-size: 10px; letter-spacing: .13em;
    text-transform: uppercase; color: var(--tenue); margin-bottom: 9px;
  }}
  .cab-via b {{ color: var(--marca); font-weight: 600; }}
  .cab-t {{
    font-size: 23px; font-weight: 600; letter-spacing: -.022em;
    line-height: 1.2; margin: 0 0 9px; color: var(--tinta);
  }}
  .cab-r {{
    font-size: 14.5px; line-height: 1.55; color: var(--suave);
    max-width: 78ch; margin: 0;
  }}
  .cab-r b {{ color: var(--tinta); font-weight: 600; }}

  /* barra de contexto: universo, periodo, corte */
  .ctx {{
    display: flex; flex-wrap: wrap; gap: 0;
    border: 1px solid var(--linea); border-radius: 4px;
    background: var(--blanco); margin-bottom: 16px; overflow: hidden;
  }}
  .ctx-i {{ padding: 9px 18px; border-right: 1px solid var(--linea); min-width: 0; }}
  .ctx-i:last-child {{ border-right: none; }}
  .ctx-k {{
    font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--tenue);
  }}
  .ctx-v {{
    font-family: var(--mono); font-size: 13px; font-weight: 500;
    color: var(--tinta); margin-top: 3px; font-variant-numeric: tabular-nums;
  }}

  /* panel de indicadores */
  .kpis {{ display: grid; gap: 1px; background: var(--linea);
          border: 1px solid var(--linea); border-radius: 4px;
          overflow: hidden; margin: 0 0 16px; }}
  .kpi {{ background: var(--blanco); padding: 13px 16px 14px; min-width: 0; }}
  .kpi-k {{
    font-family: var(--mono); font-size: 9.5px; letter-spacing: .11em;
    text-transform: uppercase; color: var(--tenue);
    display: flex; align-items: center; gap: 5px;
  }}
  .kpi-v {{
    font-family: var(--mono); font-size: 26px; font-weight: 500;
    letter-spacing: -.03em; line-height: 1.15; margin-top: 7px;
    font-variant-numeric: tabular-nums;
  }}
  .kpi-u {{ font-size: 14px; font-weight: 400; color: var(--tenue); margin-left: 2px; }}
  .kpi-d {{ font-size: 11.5px; color: var(--suave); margin-top: 6px; line-height: 1.45; }}

  /* termino con definicion */
  .term {{
    border-bottom: 1px dotted var(--tenue); cursor: help;
    text-decoration: none; color: inherit;
  }}
  .term:hover {{ border-bottom-color: var(--marca); color: var(--marca); }}

  /* secciones */
  .sec {{
    display: flex; align-items: baseline; gap: 10px;
    margin: 26px 0 3px; padding-bottom: 7px;
    border-bottom: 1px solid var(--lineaf);
  }}
  .sec-n {{
    font-family: var(--mono); font-size: 10px; letter-spacing: .13em;
    color: var(--tenue); text-transform: uppercase;
  }}
  .sec-t {{ font-size: 15.5px; font-weight: 600; letter-spacing: -.01em; color: var(--tinta); }}
  .sec-sub {{ font-size: 12.5px; color: var(--suave); margin: 9px 0 12px; max-width: 86ch; line-height: 1.55; }}

  /* tablas */
  .tcaja {{ overflow-x: auto; border: 1px solid var(--linea);
            border-radius: 4px; margin: 12px 0 16px; background: var(--blanco); }}
  table.t {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  table.t th {{
    font-size: 10px; font-weight: 600; letter-spacing: .09em;
    text-transform: uppercase; color: var(--tenue); text-align: left;
    background: var(--fondo); padding: 8px 13px 7px;
    border-bottom: 1px solid var(--lineaf); white-space: nowrap; vertical-align: bottom;
  }}
  table.t th.n {{ text-align: right; }}
  table.t th small {{
    display: block; font-weight: 400; font-size: 9.5px; color: var(--tenue);
    letter-spacing: .03em; text-transform: none; margin-top: 2px; opacity: .85;
  }}
  table.t td {{
    padding: 6px 13px; border-bottom: 1px solid var(--linea);
    white-space: nowrap; color: var(--tinta);
  }}
  table.t tbody tr:last-child td {{ border-bottom: none; }}
  table.t tbody tr:hover td {{ background: var(--fondo); }}
  table.t td.n {{
    font-family: var(--mono); font-variant-numeric: tabular-nums;
    text-align: right; font-size: 12px;
  }}
  table.t td.txt {{ white-space: normal; min-width: 190px; line-height: 1.45; }}
  table.t td.tick {{ font-family: var(--mono); font-weight: 600; font-size: 12px; }}
  table.t tr.destaca td {{ background: var(--marcac); font-weight: 600; }}
  table.t tr.destaca:hover td {{ background: var(--marcac); }}
  .sube {{ color: var(--alto); }}
  .baja {{ color: var(--bajo); }}
  .pildora {{
    display: inline-block; font-family: var(--mono); font-size: 9.5px;
    font-weight: 600; letter-spacing: .06em; padding: 1.5px 6px;
    border-radius: 2px; text-transform: uppercase;
  }}
  .p-si {{ background: var(--altoc); color: var(--alto); }}
  .p-no {{ background: var(--fondo); color: var(--tenue); }}

  /* notas */
  .nota {{
    border: 1px solid var(--linea); border-left: 3px solid var(--nc, var(--suave));
    background: var(--nf, var(--blanco)); padding: 13px 16px;
    border-radius: 4px; margin: 14px 0; font-size: 13px; line-height: 1.58;
  }}
  .nota-t {{
    font-family: var(--mono); font-size: 9.5px; font-weight: 600;
    letter-spacing: .13em; text-transform: uppercase;
    color: var(--nc, var(--suave)); display: block; margin-bottom: 6px;
  }}
  .nota p {{ margin: 0 0 8px; }} .nota p:last-child {{ margin-bottom: 0; }}

  /* detalle metodologico */
  details.det {{ border: 1px solid var(--linea); border-radius: 4px;
                 margin: 14px 0; background: var(--blanco); }}
  details.det > summary {{
    cursor: pointer; padding: 10px 14px; font-size: 11.5px; font-weight: 600;
    color: var(--marca); list-style: none; display: flex; align-items: center;
    gap: 7px; letter-spacing: .03em; text-transform: uppercase;
    font-family: var(--mono);
  }}
  details.det > summary::-webkit-details-marker {{ display: none; }}
  details.det > summary::before {{ content: "▸"; font-size: 10px; }}
  details.det[open] > summary::before {{ content: "▾"; }}
  details.det[open] > summary {{ border-bottom: 1px solid var(--linea); }}
  .det-c {{ padding: 14px 16px; font-size: 13px; line-height: 1.62; }}
  .det-c p {{ margin: 0 0 10px; }} .det-c p:last-child {{ margin-bottom: 0; }}
  .det-c ul {{ margin: 0 0 10px; padding-left: 18px; }}
  .det-c li {{ margin-bottom: 5px; }}
  code {{ font-family: var(--mono); font-size: .9em; background: var(--fondo);
          padding: 1px 4px; border-radius: 2px; }}
  .fx {{
    font-family: var(--mono); font-size: 12.5px; background: var(--fondo);
    border: 1px solid var(--linea); border-radius: 3px; padding: 12px 14px;
    margin: 11px 0; overflow-x: auto; white-space: pre; line-height: 1.75;
  }}

  /* controles */
  .block-container label {{
    font-size: 10.5px !important; font-weight: 600 !important;
    letter-spacing: .09em !important; text-transform: uppercase !important;
    color: var(--tenue) !important; white-space: normal !important;
    line-height: 1.35 !important;
  }}
  div[data-testid="stMetricValue"] {{ font-family: var(--mono); }}

  @media (max-width: 760px) {{
    .block-container {{ padding: .8rem 1rem 3rem; }}
    .cab-t {{ font-size: 19px; }}
    .kpi-v {{ font-size: 21px; }}
  }}
</style>
    """)
    _tema_graficas()


# ==========================================================================
# 2. Glosario
# ==========================================================================
# Cada termino tecnico que aparece en la interfaz vive aqui con su definicion
# en lenguaje normal. Conservar el termino correcto y explicarlo es mejor que
# sustituirlo: el usuario se va a topar con el mismo nombre en cualquier otro
# sistema, y cambiarlo solo lo deja sin saber como se llama lo que vio.
GLOSARIO = {
    "beta": "Cuanto amplifica una accion los movimientos del mercado. Beta 1.3 significa que si el mercado cae 1%, esa accion tiende a caer 1.3%.",
    "beta condicional": "La beta medida por separado en cada regimen de mercado, en vez de un solo numero para toda la historia.",
    "volatilidad": "Que tanto se mueve un precio, hacia arriba o hacia abajo. Se expresa anualizada: 20% quiere decir que en un anio tipico se mueve esa magnitud.",
    "volatilidad condicional": "La volatilidad que el modelo estima para un dia concreto, usando solo lo que pasó antes de ese dia.",
    "correlacion": "Que tanto se mueven dos acciones al mismo tiempo. Va de -1 a +1. Cero significa que una no dice nada de la otra.",
    "rendimiento logaritmico": "El cambio de precio de un dia al siguiente, en escala logaritmica. Se usa en vez del precio porque dos acciones que solo suben con los anios darian correlacion alta sin tener relacion.",
    "covarianza": "La tabla que guarda como se mueve cada par de activos entre si. Es el insumo del calculo de riesgo de una cartera.",
    "contribucion al riesgo": "Que parte del riesgo total de la cartera aporta cada posicion. No es proporcional al dinero invertido: una posicion del 5% puede aportar el 30% del riesgo.",
    "ratio de diversificacion": "Cuanto riesgo se cancela al juntar las posiciones. 1.00 significa que no se cancela nada; mas alto es mejor.",
    "minima varianza": "La combinacion de pesos que produce la cartera menos volatil posible con esos activos.",
    "encogimiento": "Tecnica (Ledoit-Wolf) que mezcla la covarianza observada con una version mas estable. Evita que la optimizacion persiga ruido cuando hay pocos datos por activo.",
    "GARCH": "Modelo que predice volatilidad aprovechando que la volatilidad viene en rachas: despues de un dia movido es mas probable que venga otro.",
    "persistencia": "Cuanto duran esas rachas. Cerca de 0.98 significa que despues de un susto el mercado tarda semanas en calmarse.",
    "validacion hacia adelante": "Metodo de prueba que simula ser un inversionista del pasado: solo usa informacion anterior a la fecha que esta evaluando, nunca el futuro.",
    "rotacion": "Que porcentaje de la cartera hay que mover en cada rebalanceo. Mas rotacion significa mas costos de operacion.",
    "dependencia de cola": "La probabilidad de que dos activos esten simultaneamente en sus peores dias. Mide si caen juntos, que es lo que la correlacion promedio no distingue.",
    "informacion mutua": "Medida de dependencia que detecta cualquier relacion, no solo las lineales. Vale cero unicamente si los dos activos son independientes.",
    "cofactor": "Variable que modifica el efecto de otra en vez de sumar el suyo propio. Entra en el modelo como producto.",
    "regimen": "Estado del mercado. Aqui son dos: calma y tension, definidos por la volatilidad que el modelo GARCH estima cada dia.",
    "significativo": "Que el resultado es lo bastante grande frente a su margen de error como para no atribuirlo al azar. El corte usado es el 5%.",
    "error estandar": "El margen de error de una estimacion. Mientras mas chico, mas confiable es el numero.",
    "error estandar robusto": "Margen de error calculado sin suponer que la varianza es constante. Hace falta aqui porque el propio GARCH demostro que no lo es.",
    "R cuadrado": "Que fraccion del movimiento de una accion queda explicada por el mercado. 0.30 significa 30%; el resto son razones propias de la empresa.",
    "quintil": "Division en cinco grupos del mismo tamanio, ordenados de menor a mayor.",
}


def t(clave: str, texto: str | None = None) -> str:
    """Marca un termino tecnico con su definicion al pasar el cursor.

    Devuelve HTML, asi que se usa dentro de los componentes de abajo y no con
    ``st.write``. El termino se conserva tal cual; lo que se agrega es el
    acceso a su significado.
    """
    definicion = GLOSARIO.get(clave, "")
    visible = texto or clave
    if not definicion:
        return visible
    return f'<span class="term" title="{definicion}">{visible}</span>'


# ==========================================================================
# 3. Componentes
# ==========================================================================
def cabecera(via: str, titulo: str, respuesta: str) -> None:
    """Cabecera de la vista: ruta, pregunta y la respuesta en una frase.

    La respuesta va arriba de todo, antes de cualquier grafica, porque quien
    abre la vista no deberia tener que deducirla de los numeros.
    """
    html(
        f'<div class="cab"><div class="cab-via">{via}</div>'
        f'<div class="cab-t">{titulo}</div>'
        f'<p class="cab-r">{respuesta}</p></div>',
    )


def contexto(items: list[tuple[str, str]]) -> None:
    """Barra de procedencia: universo, periodo, fecha de corte, metodo."""
    celdas = "".join(
        f'<div class="ctx-i"><div class="ctx-k">{k}</div>'
        f'<div class="ctx-v">{v}</div></div>' for k, v in items
    )
    html(f'<div class="ctx">{celdas}</div>')


def kpis(items: list[dict]) -> None:
    """Panel de indicadores. Cada uno lleva unidad y una linea de lectura.

    ``items`` acepta ``clave``, ``valor``, ``unidad``, ``detalle``, ``tono`` y
    ``termino`` (una clave del glosario, que vuelve la etiqueta consultable).
    """
    celdas = ""
    for it in items:
        color, _, _ = TONOS.get(it.get("tono", "neutro"), TONOS["neutro"])
        etiqueta = t(it["termino"], it["clave"]) if it.get("termino") else it["clave"]
        unidad = f'<span class="kpi-u">{it["unidad"]}</span>' if it.get("unidad") else ""
        det = f'<div class="kpi-d">{it["detalle"]}</div>' if it.get("detalle") else ""
        celdas += (
            f'<div class="kpi"><div class="kpi-k">{etiqueta}</div>'
            f'<div class="kpi-v" style="color:{color}">{it["valor"]}{unidad}</div>'
            f"{det}</div>"
        )
    html(
        f'<div class="kpis" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">'
        f"{celdas}</div>",
    )


def seccion(numero: str, titulo: str, subtitulo: str = "") -> None:
    sub = f'<p class="sec-sub">{subtitulo}</p>' if subtitulo else ""
    html(
        f'<div class="sec"><span class="sec-n">{numero}</span>'
        f'<span class="sec-t">{titulo}</span></div>{sub}',
    )


def nota(cuerpo: str, titulo: str = "", tono: str = "neutro") -> None:
    color, _, fondo = TONOS.get(tono, TONOS["neutro"])
    enc = f'<span class="nota-t">{titulo}</span>' if titulo else ""
    html(
        f'<div class="nota" style="--nc:{color};--nf:{fondo}">{enc}{cuerpo}</div>',
    )


def detalle(titulo: str, cuerpo: str) -> None:
    """Metodologia plegada: formulas, umbrales y supuestos."""
    html(
        f'<details class="det"><summary>{titulo}</summary>'
        f'<div class="det-c">{cuerpo}</div></details>',
    )


def tabla(
    df: pd.DataFrame,
    columnas: dict[str, dict],
    destaca=None,
    clases: dict | None = None,
) -> None:
    """Tabla con encabezados en dos niveles: el nombre y su unidad o aclaracion.

    ``columnas`` mapea la columna del DataFrame a
    ``{"nombre", "nota", "fmt", "num", "tick", "termino"}``. La ``nota`` es la
    linea chica bajo el encabezado, y es obligatoria en la practica: una
    columna cuyo nombre no se entiende sin contexto no deberia mostrarse.
    """
    th = ""
    for conf in columnas.values():
        nombre = t(conf["termino"], conf["nombre"]) if conf.get("termino") else conf["nombre"]
        sub = f'<small>{conf["nota"]}</small>' if conf.get("nota") else ""
        th += f'<th class="{"n" if conf.get("num") else ""}">{nombre}{sub}</th>'

    filas = ""
    for _, fila in df.iterrows():
        cls = ' class="destaca"' if destaca is not None and destaca(fila) else ""
        tds = ""
        for col, conf in columnas.items():
            v = fila[col]
            txt = conf["fmt"](v) if conf.get("fmt") else str(v)
            tipo = "n" if conf.get("num") else ("tick" if conf.get("tick") else "txt")
            extra = f" {clases[col](v)}" if clases and col in clases else ""
            tds += f'<td class="{tipo}{extra}">{txt}</td>'
        filas += f"<tr{cls}>{tds}</tr>"

    html(
        f'<div class="tcaja"><table class="t"><thead><tr>{th}</tr></thead>'
        f"<tbody>{filas}</tbody></table></div>",
    )


def pildora(valor: bool, si: str = "SÍ", no: str = "NO") -> str:
    return f'<span class="pildora {"p-si" if valor else "p-no"}">{si if valor else no}</span>'


# ==========================================================================
# 4. Graficas
# ==========================================================================
def _tema_graficas() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": BLANCO,
        "axes.facecolor": BLANCO,
        "axes.edgecolor": LINEA_F,
        "axes.linewidth": .8,
        "axes.labelcolor": SUAVE,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlecolor": TINTA,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": LINEA,
        "grid.linewidth": .7,
        "xtick.color": TENUE,
        "ytick.color": TENUE,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.autolayout": True,
    })


def lienzo(alto: float = 3.2, ancho: float = 9.0):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    return fig, ax


def mostrar(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)
