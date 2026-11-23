"""Terminal de correlacion y riesgo · S&P 500.

Responde el ultimo requisito del Entregable 3, "que el usuario tenga donde
verlo", y sirve de recorrido por todo el proyecto.

    streamlit run dashboard/app.py

Como esta organizado
--------------------
Cada vista responde **una** pregunta y la contesta en su cabecera, antes de
mostrar un solo numero. Despues vienen los indicadores, luego el detalle en
tablas, y al final la metodologia plegada.

La terminologia tecnica se conserva -beta, volatilidad condicional,
contribucion al riesgo- porque es la correcta y la que el usuario va a
encontrar en cualquier otro sistema. Lo que no se acepta es que aparezca sin
explicacion: cada termino esta marcado con su definicion, cada columna lleva
su unidad, y cada metodo esta documentado en su bloque plegable.

Todo el analisis vive en ``src/``, que es la capa con las 307 pruebas. Este
archivo solo presenta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import datos as D
import estilo as E
from src.risk import DIAS_HABILES

st.set_page_config(page_title="Riesgo S&P 500 · Terminal",
                   page_icon="◧", layout="wide")
E.aplicar_estilo()

pct0 = lambda v: f"{v:.0%}"
pct1 = lambda v: f"{v:.1%}"
pct2 = lambda v: f"{v:.2%}"
num2 = lambda v: f"{v:.2f}"
num3 = lambda v: f"{v:.3f}"
sgn3 = lambda v: f"{v:+.3f}"
ent = lambda v: f"{int(v):,}"


def direccion(v) -> str:
    return "sube" if v > 0 else ("baja" if v < 0 else "")


# ==========================================================================
# Resumen ejecutivo
# ==========================================================================
def v_resumen():
    rets, _, _ = D.cargar()
    _, par, vol, tension = D.mercado_y_garch("v1")

    E.cabecera(
        "Resumen ejecutivo",
        "Estado del riesgo del universo",
        "El motor mide cómo se mueven juntas 457 acciones del S&P 500 y traduce "
        "eso al riesgo de una cartera. Las cuatro conclusiones del proyecto están "
        "abajo; cada vista del menú desarrolla una.",
    )

    E.contexto([
        ("Universo", f"{rets.shape[1]} activos"),
        ("Periodo", f"{rets.index[0]:%Y-%m-%d} → {rets.index[-1]:%Y-%m-%d}"),
        ("Observaciones", f"{rets.shape[0]:,} días"),
        ("Pares medibles", f"{rets.shape[1] * (rets.shape[1] - 1) // 2:,}"),
        ("Fuente", "Yahoo Finance · ajustado"),
    ])

    actual, largo = vol.iloc[-1], par["vol_largo_plazo"]
    E.kpis([
        {"clave": "Volatilidad del mercado", "termino": "volatilidad condicional",
         "valor": f"{actual:.1%}", "tono": "bajo" if actual < largo else "alto",
         "detalle": f"Nivel de largo plazo {largo:.1%}. "
                    f"{'Por debajo' if actual < largo else 'Por encima'} en "
                    f"{abs(actual - largo):.1%} puntos."},
        {"clave": "Persistencia", "termino": "persistencia",
         "valor": f"{par['persistencia']:.3f}",
         "detalle": "Cerca de 1: tras un sobresalto, el mercado tarda semanas en calmarse."},
        {"clave": "Días en tensión", "valor": f"{tension.sum():,}",
         "termino": "regimen",
         "detalle": f"{tension.mean():.0%} del periodo. Quintil superior de volatilidad."},
        {"clave": "Correlación media", "termino": "correlacion",
         "valor": "0.33",
         "detalle": "Entre pares del universo. Sube en los episodios de tensión."},
    ])

    E.seccion("01", "Conclusiones",
              "Cada una se desarrolla en su propia vista, con el método y los datos que la sustentan.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        E.nota(
            "<p>Dos acciones del mismo sector se mueven juntas <b>1.39 veces más</b> "
            "que dos de sectores distintos (0.4404 contra 0.3164 de correlación "
            "media). Repartir capital entre cinco bancos no reparte el riesgo.</p>",
            "01 · Un sector no diversifica", "marca")
        E.nota(
            "<p>Probando ventanas de 125 a 2,000 días con validación hacia adelante, "
            "la que produce la cartera menos volátil es la de <b>250 días</b>. "
            "Más historia arrastra un mercado que ya no existe.</p>",
            "03 · Un año de historia basta", "marca")
    with c2:
        E.nota(
            "<p>La probabilidad de que dos acciones estén juntas en su peor 10% es "
            "<b>29.9%</b>, contra 26.8% de estar juntas en su mejor 10%. Por azar "
            "serían 10% ambas. Ocurre en el <b>84.5%</b> de los pares.</p>",
            "02 · La cobertura falla a la baja", "alto")
        E.nota(
            "<p>El <b>51.6%</b> de las acciones cambia su sensibilidad al mercado de "
            "forma significativa al entrar en tensión, y lo hace <b>hacia 1</b>. Las "
            "defensivas dejan de serlo: Utilities pasa de beta 0.479 a 0.779.</p>",
            "04 · Las betas convergen en crisis", "alto")

    E.seccion("02", "Volatilidad condicional del mercado",
              "Estimada día a día con un modelo GARCH(1,1) sobre el índice equiponderado "
              "del universo. El modelo no conoce ningún calendario de eventos: el pico de "
              "marzo de 2020 lo produce únicamente a partir de los precios.")

    fig, ax = E.lienzo(3.0)
    ax.plot(vol.index, vol * 100, color=E.MARCA, linewidth=.85)
    ax.fill_between(vol.index, 0, vol * 100, where=tension.to_numpy(),
                    color=E.ALTO, alpha=.13, linewidth=0, label="régimen de tensión")
    ax.axhline(largo * 100, color=E.SUAVE, linestyle=(0, (4, 3)), linewidth=.9,
               label=f"largo plazo · {largo:.1%}")
    ax.set_ylabel("volatilidad anualizada (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncols=2)
    E.mostrar(fig)


# ==========================================================================
# Estructura de correlación
# ==========================================================================
def v_correlacion(n: int):
    pares, comp = D.pares_y_sectores(n)
    mismo = comp.loc["Mismo sector", "media"]
    distinto = comp.loc["Distinto sector", "media"]

    E.cabecera(
        "Entregable 1 · Estructura de correlación",
        "¿Las acciones del mismo sector se mueven juntas?",
        f"Sí, y la diferencia es grande. Las del mismo sector promedian "
        f"<b>{mismo:.4f}</b> de correlación contra <b>{distinto:.4f}</b> las de "
        f"sectores distintos, <b>{mismo / distinto:.2f} veces más</b>. "
        f"Consecuencia operativa: concentrar en un sector no diversifica.",
    )

    E.contexto([
        ("Muestra", f"{n} activos"),
        ("Pares", f"{len(pares):,}"),
        ("Método", "Pearson sobre rendimientos log"),
        ("Universo completo", "503 activos → 1.39×"),
    ])

    E.kpis([
        {"clave": "Mismo sector", "termino": "correlacion", "valor": f"{mismo:.4f}",
         "detalle": f"{int(comp.loc['Mismo sector', 'n_pares']):,} pares. "
                    f"Mediana {comp.loc['Mismo sector', 'mediana']:.4f}."},
        {"clave": "Sectores distintos", "termino": "correlacion", "valor": f"{distinto:.4f}",
         "detalle": f"{int(comp.loc['Distinto sector', 'n_pares']):,} pares. "
                    f"Mediana {comp.loc['Distinto sector', 'mediana']:.4f}."},
        {"clave": "Razón", "valor": f"{mismo / distinto:.2f}", "unidad": "×",
         "tono": "marca", "detalle": "Cuánto más se parecen dentro del sector."},
        {"clave": "Dispersión intra", "valor": f"{comp.loc['Mismo sector', 'desv_est']:.4f}",
         "detalle": "Desviación estándar dentro del grupo. Más alta: el sector no es homogéneo."},
    ])

    E.detalle("Por qué se correlacionan rendimientos y no precios", """
        <p>Dos acciones que simplemente subieron durante once años producirían una
        correlación alta aunque no tengan ninguna relación económica: las dos crecen
        con el tiempo, y la correlación detecta esa tendencia compartida. Es el caso
        clásico de <b>correlación espuria</b> por series no estacionarias.</p>
        <p>El rendimiento logarítmico <code>ln(P_t / P_t−1)</code> elimina el nivel y
        deja solo el movimiento del día, que es lo comparable. Los precios además se
        toman <b>ajustados por splits y dividendos</b>: sin ese ajuste, un split 4:1
        aparecería como una caída de −75% en un día y contaminaría toda la matriz.</p>
    """)

    E.seccion("01", "Distribución de las correlaciones por pares")
    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        fig, ax = E.lienzo(3.1, 7)
        for flag, color, etq in [(True, E.MARCA, "mismo sector"), (False, E.TENUE, "sectores distintos")]:
            sub = pares[pares["same_sector"] == flag]["corr"]
            ax.hist(sub, bins=45, alpha=.72, color=color, density=True,
                    label=f"{etq} · n={len(sub):,}", edgecolor="none")
        ax.axvline(mismo, color=E.MARCA, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.axvline(distinto, color=E.SUAVE, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.set_xlabel("coeficiente de correlación de Pearson")
        ax.set_ylabel("densidad")
        ax.legend(loc="upper right")
        E.mostrar(fig)
    with der:
        E.html('<div class="sec-sub" style="margin-top:4px">'
                    'Pares con mayor correlación de la muestra.</div>',
                    )
        E.tabla(
            pares.nlargest(9, "corr"),
            {
                "ticker_a": {"nombre": "A", "tick": True},
                "ticker_b": {"nombre": "B", "tick": True},
                "corr": {"nombre": "ρ", "nota": "Pearson", "fmt": num3, "num": True,
                         "termino": "correlacion"},
                "sector_a": {"nombre": "Sector"},
            },
        )

    mostrados = min(n, 60)
    E.seccion("02", "Matriz de correlación ordenada por sector",
              "Los tickers se ordenan por sector, de modo que cada sector forma un bloque "
              "contiguo. Los cuadros oscuros sobre la diagonal son esos bloques. La "
              "diagonal propia se omite: vale 1 por definición y comprimiría la escala. "
              + (f"Se muestran {mostrados} de los {n} activos de la muestra: por encima "
                 f"de esa cifra las etiquetas dejan de ser legibles, aunque las cifras "
                 f"de arriba sí usan la muestra completa."
                 if n > mostrados else
                 f"Se muestran los {mostrados} activos de la muestra."))

    matriz = D.matriz_por_sector(mostrados)
    valores = matriz.to_numpy().astype(float)
    np.fill_diagonal(valores, np.nan)
    lo, hi = np.nanpercentile(valores, [2, 98])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7.4))
    im = ax.imshow(valores, cmap="RdYlBu_r", vmin=lo, vmax=hi)
    ax.set_xticks(range(len(matriz)), matriz.columns, rotation=90, fontsize=5.2)
    ax.set_yticks(range(len(matriz)), matriz.index, fontsize=5.2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(E.LINEA_F)
    cb = fig.colorbar(im, ax=ax, shrink=.72, pad=.015)
    cb.set_label("correlación", size=9)
    cb.outline.set_edgecolor(E.LINEA_F)
    E.mostrar(fig)


# ==========================================================================
# Red del mercado
# ==========================================================================
def v_red(n: int):
    r = D.red(n)
    arbol, pur = r["arbol"], r["pureza"]
    sectores, pos = r["sectores"], r["posiciones"]

    E.cabecera(
        "Entregable 1 · Estructura de la red",
        "¿Qué conexiones sostienen al mercado?",
        f"De las <b>{r['n_posibles']:,}</b> correlaciones posibles, bastan "
        f"<b>{len(arbol)}</b> para mantener conectado a todo el universo. De esas, "
        f"el <b>{pur['observado']:.1%}</b> une activos del mismo sector, frente al "
        f"<b>{pur['esperado_por_azar']:.1%}</b> que caería ahí por azar: "
        f"<b>{pur['razon']:.1f} veces más</b>.",
    )

    E.contexto([
        ("Activos", f"{len(pos)}"),
        ("Correlaciones posibles", f"{r['n_posibles']:,}"),
        ("Aristas del esqueleto", f"{len(arbol)}"),
        ("Método", "Árbol mínimo · Mantegna (1999)"),
    ])

    E.kpis([
        {"clave": "Dentro del mismo sector", "valor": f"{pur['observado']:.1%}",
         "tono": "marca",
         "detalle": f"{pur['mismo_sector']} de {pur['aristas']} aristas."},
        {"clave": "Esperado por azar", "valor": f"{pur['esperado_por_azar']:.1%}",
         "detalle": "Dos activos al azar comparten sector con esta probabilidad."},
        {"clave": "Razón", "valor": f"{pur['razon']:.1f}", "unidad": "×",
         "tono": "marca",
         "detalle": "El Entregable 1 midió 1.39× sobre la correlación promedio."},
        {"clave": "Mayor número de conexiones", "valor": f"{r['grados']['grado'].max()}",
         "detalle": f"{r['grados'].iloc[0]['ticker']}. El promedio en un árbol es 2."},
    ])

    E.nota(
        f"<p>Es la misma pregunta del Entregable 1, hecha de forma más exigente. "
        f"Un promedio alto puede salir de muchas relaciones tibias; aquí solo "
        f"caben las <b>{len(arbol)} conexiones más fuertes</b> del mercado, las "
        f"mínimas que lo mantienen unido sin ciclos.</p>"
        f"<p>El efecto sectorial pasa de <b>1.39×</b> sobre la correlación "
        f"promedio a <b>{pur['razon']:.1f}×</b> sobre el esqueleto. El sector no "
        f"solo inclina el promedio: <b>organiza la estructura</b>.</p>",
        "Por qué esto es más fuerte que la correlación promedio", "marca")

    E.seccion("01", "El esqueleto del mercado",
              "Cada punto es un activo y su color el sector. Las posiciones las "
              "calcula un método de fuerzas: los conectados se atraen y todos se "
              "repelen. Nadie le dice al algoritmo que existen los sectores, así "
              "que si aparecen agrupados, es un hallazgo y no una decisión de "
              "presentación.")

    import matplotlib.pyplot as plt

    lista_sectores = sorted({s for s in sectores.values() if s})
    mapa = plt.get_cmap("tab20")
    color_de = {s: mapa(i % 20) for i, s in enumerate(lista_sectores)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for f in arbol.itertuples():
        a, b = pos.loc[f.origen], pos.loc[f.destino]
        mismo = sectores.get(f.origen) == sectores.get(f.destino)
        ax.plot([a.x, b.x], [a.y, b.y],
                color=color_de.get(sectores.get(f.origen), E.TENUE) if mismo else E.LINEA_F,
                linewidth=1.1 if mismo else .6,
                alpha=.75 if mismo else .5, zorder=1)

    grado = r["grados"].set_index("ticker")["grado"]
    ax.scatter(pos["x"], pos["y"],
               s=14 + 9 * grado.reindex(pos.index).fillna(1),
               c=[color_de.get(sectores.get(t), E.TENUE) for t in pos.index],
               edgecolors="white", linewidths=.5, zorder=2)

    for ticker in r["grados"].head(10)["ticker"]:
        if ticker in pos.index:
            ax.annotate(ticker, (pos.loc[ticker, "x"], pos.loc[ticker, "y"]),
                        fontsize=7.5, fontweight="600", color=E.TINTA,
                        xytext=(4, 4), textcoords="offset points", zorder=3)

    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color(E.LINEA)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                                  color=color_de[s], label=s)
                       for s in lista_sectores],
              loc="center left", bbox_to_anchor=(1.0, .5), fontsize=8)
    E.mostrar(fig)
    E.html('<div class="sec-sub">El tamaño del punto es el número de '
                'conexiones. Las líneas grises cruzan sectores; las de color se '
                'quedan dentro de uno. Se etiquetan los diez concentradores '
                'principales.</div>')

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        E.seccion("02", "Las conexiones más fuertes",
                  "Las aristas de menor distancia del esqueleto.")
        tabla = arbol.head(12).copy()
        tabla["sector"] = tabla["origen"].map(sectores).fillna("")
        tabla["mismo"] = [sectores.get(a) == sectores.get(b)
                          for a, b in zip(tabla["origen"], tabla["destino"])]
        E.tabla(tabla, {
            "origen": {"nombre": "A", "tick": True},
            "destino": {"nombre": "B", "tick": True},
            "correlacion": {"nombre": "ρ", "fmt": num3, "num": True,
                            "termino": "correlacion"},
            "distancia": {"nombre": "Distancia", "nota": "Mantegna",
                          "fmt": num3, "num": True},
            "sector": {"nombre": "Sector de A"},
            "mismo": {"nombre": "Mismo", "fmt": lambda v: E.pildora(v, "SÍ", "no")},
        })
        E.nota(
            "<p>Las dos aristas más fuertes del S&P 500 son <b>GOOG–GOOGL</b> y "
            "<b>NWSA–NWS</b>: las clases de acción A y C de Alphabet y de News "
            "Corp. Son la misma empresa, así que su correlación cercana a 1 no "
            "es información sobre el mercado.</p>"
            "<p>El método las encuentra solo, sin que nadie le avise, y eso "
            "confirma que el esqueleto captura lo que debe. El README ya las "
            "listaba como limitación del Entregable 1.</p>",
            "Control de cordura", "aviso")
    with der:
        E.seccion("03", "Concentradores",
                  "Activos que el mercado usa como punto de paso.")
        E.tabla(r["grados"].head(12), {
            "ticker": {"nombre": "Activo", "tick": True},
            "sector": {"nombre": "Sector"},
            "grado": {"nombre": "Conexiones", "fmt": ent, "num": True},
        })
        E.nota(
            "<p>En un árbol el grado promedio es 2 y casi todo son hojas. Un "
            "concentrador no es el activo más volátil ni el más correlacionado "
            "con el índice: es el que, si se mueve, <b>arrastra a más vecinos "
            "directos</b>.</p>", "", "marca")

    E.seccion("04", "Qué tan compacto es cada sector",
              "Proporción de las aristas que tocan al sector y se quedan dentro "
              "de él. Alta: bloque cerrado. Baja: repartido por la red.")
    por_sector = r["por_sector"]
    izq, der = st.columns([2, 3], gap="medium")
    with izq:
        fig, ax = E.lienzo(3.8, 5.2)
        E.barras(ax, por_sector["sector"], por_sector["proporcion_interna"] * 100,
                 colores=[E.MARCA] * len(por_sector), fmt="{:.0f}%")
        ax.set_xlabel("aristas internas (%)")
        E.mostrar(fig)
    with der:
        E.tabla(por_sector, {
            "sector": {"nombre": "Sector"},
            "aristas": {"nombre": "Aristas", "nota": "que lo tocan",
                        "fmt": ent, "num": True},
            "internas": {"nombre": "Internas", "fmt": ent, "num": True},
            "proporcion_interna": {"nombre": "Proporción", "fmt": pct0, "num": True},
            "distancia_media": {"nombre": "Distancia media", "fmt": num3, "num": True},
        })

    E.detalle("Qué es un árbol de expansión mínima y por qué este", """
        <p><b>El problema.</b> Con 458 activos hay 104,653 correlaciones. La matriz
        las contiene todas y por eso no se puede leer: cada activo está conectado con
        todos los demás, así que no hay estructura visible, solo una nube de números.</p>
        <p><b>La solución.</b> El árbol de expansión mínima se queda con la cantidad
        mínima de conexiones que mantiene unido a todo el mercado: exactamente n−1
        aristas, las más cortas posibles, sin ciclos. Las que sobreviven son las que
        de verdad sostienen la estructura.</p>
        <p><b>La distancia.</b> La correlación no sirve como distancia tal cual: ρ=1
        significa "idénticos" y una distancia debe valer 0 ahí. La métrica de Mantegna
        (1999) invierte la escala y además cumple la desigualdad del triángulo, que es
        lo que permite meterla a un árbol:</p>
        <div class="fx">d(i,j) = √(2 · (1 − ρ(i,j)))

  ρ = +1 → d = 0        ρ = 0 → d ≈ 1.414        ρ = −1 → d = 2</div>
        <p><b>El algoritmo.</b> Prim, implementado a mano en quince líneas: arranca en
        un activo y en cada paso agrega la arista más corta que conecta lo que ya está
        dentro con algo que falta. El resultado no depende del punto de partida
        mientras no haya empates exactos, cosa que con datos reales no ocurre.</p>
        <p><b>La referencia contra el azar.</b> El porcentaje observado por sí solo no
        dice nada, porque algunas aristas caerían dentro del mismo sector aunque se
        repartieran al azar: los sectores grandes tienen más parejas posibles. La
        referencia es la probabilidad de que dos activos distintos al azar compartan
        sector, <code>Σ n_s(n_s−1) / n(n−1)</code>, y es la que aparece en el panel de
        arriba junto al dato observado.</p>
    """)


# ==========================================================================
# Descomposición de riesgo
# ==========================================================================
def v_riesgo(cartera: dict):
    _, _, sectores = D.cargar()
    sel, metodo, encoger = cartera["posiciones"], cartera["metodo"], cartera["encoger"]

    E.cabecera(
        "Entregable 2 · Motor de riesgo",
        "¿Cuánto riesgo tiene esta cartera y de dónde viene?",
        "Selecciona posiciones y el motor calcula la volatilidad de la cartera, "
        "qué parte del riesgo aporta cada una y qué pesos lo minimizarían. "
        "La contribución al riesgo <b>no es proporcional al capital invertido</b>, "
        "y esa diferencia es lo que el motor existe para mostrar.",
    )

    if len(sel) < 2:
        E.nota("<p>Define una cartera de al menos dos posiciones en la barra lateral: "
               "con una sola no hay covarianza que descomponer.</p>",
               "Cartera insuficiente", "aviso")
        return

    r, delta = D.riesgo(tuple(sel), metodo, encoger)

    E.contexto([
        ("Posiciones", str(len(sel))),
        ("Observaciones", f"{len(D.cargar()[0]):,} días"),
        ("Asignación", metodo),
        ("Estimador", f"Ledoit-Wolf δ={delta:.3f}" if encoger else "Covarianza muestral"),
    ])

    E.kpis([
        {"clave": "Volatilidad de la cartera", "termino": "volatilidad",
         "valor": f"{r['portfolio_vol']:.2%}", "tono": "marca",
         "detalle": "Anualizada. Es lo que realmente se mueve el conjunto."},
        {"clave": "Suma ponderada individual", "valor": f"{r['weighted_avg_vol']:.2%}",
         "detalle": "Lo que se movería si nada se cancelara entre posiciones."},
        {"clave": "Ratio de diversificación", "termino": "ratio de diversificacion",
         "valor": f"{r['diversification_ratio']:.3f}",
         "detalle": f"Máximo alcanzable con {len(sel)} activos: "
                    f"{r['max_diversification_ratio']:.3f} (raíz de n)."},
        {"clave": "Riesgo evitado", "valor": f"{1 - r['portfolio_vol'] / r['weighted_avg_vol']:.1%}",
         "tono": "bajo",
         "detalle": "Reducción lograda porque las posiciones no se mueven al unísono."},
    ])

    detalle_tabla = pd.DataFrame({
        "activo": r["labels"],
        "sector": [sectores.get(x, "") for x in r["labels"]],
        "peso": r["weights"],
        "vol": r["standalone_vol"],
        "riesgo": r["pct_risk"],
        "razon": r["pct_risk"] / np.where(r["weights"] > 0, r["weights"], np.nan),
    }).sort_values("riesgo", ascending=False)

    E.seccion("01", "Descomposición por posición",
              "La contribución al riesgo reparte exactamente la volatilidad total: las "
              "columnas suman 100% por el teorema de Euler, no por aproximación.")

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        E.tabla(
            detalle_tabla,
            {
                "activo": {"nombre": "Activo", "tick": True},
                "sector": {"nombre": "Sector"},
                "peso": {"nombre": "Peso", "nota": "% capital", "fmt": pct2, "num": True},
                "vol": {"nombre": "Vol. individual", "nota": "anualizada",
                        "fmt": pct1, "num": True, "termino": "volatilidad"},
                "riesgo": {"nombre": "Contribución", "nota": "% del riesgo", "fmt": pct2,
                           "num": True, "termino": "contribucion al riesgo"},
                "razon": {"nombre": "Riesgo / peso", "nota": "×", "fmt": num2, "num": True},
            },
            clases={"razon": lambda v: direccion(v - 1) if pd.notna(v) else ""},
        )
        E.html(
            f'<div class="sec-sub">Las contribuciones suman '
            f'{r["pct_risk"].sum():.4%} con un error de {r["check_sum"]:.1e}.</div>',
            )
    with der:
        fig, ax = E.lienzo(max(2.6, .34 * len(sel) + 1.1), 5.6)
        y = np.arange(len(detalle_tabla))
        ax.barh(y - .19, detalle_tabla["peso"] * 100, height=.37,
                color=E.MARCA, label="peso")
        ax.barh(y + .19, detalle_tabla["riesgo"] * 100, height=.37,
                color=E.ALTO, label="contribución al riesgo")
        ax.set_yticks(y, detalle_tabla["activo"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("%")
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        E.mostrar(fig)

    if metodo == D.MINIMA_VARIANZA:
        E.nota(
            "<p>En una cartera de mínima varianza la columna <b>riesgo / peso</b> vale "
            "1.00 en todas las posiciones con peso. No es un error de cálculo: es la "
            "<b>condición de optimalidad</b>. Si alguna posición aportara riesgo "
            "desproporcionado a su peso, mover capital fuera de ella reduciría la "
            "volatilidad total, y entonces la cartera no sería el mínimo.</p>"
            "<p>Cambia la asignación a <b>Equiponderado</b> para ver la columna con "
            "valores dispares, que es el caso general.</p>",
            "Lectura de la columna riesgo / peso", "marca")

    E.detalle("Estimador de covarianza y restricción de posiciones largas", """
        <p><b>Encogimiento de Ledoit-Wolf (2003).</b> Con N activos hay N(N+1)/2
        parámetros de covarianza que estimar y solo T observaciones. Con 1,200 activos
        y 2,900 días son 720,000 incógnitas contra 2,900 datos: la matriz muestral
        queda tan mal condicionada que invertirla amplifica el ruido. El estimador
        promedia la matriz observada con un objetivo estructurado y estable:</p>
        <div class="fx">Σ = δ · F + (1 − δ) · S</div>
        <p>El objetivo F conserva la volatilidad real de cada activo pero asume
        correlación constante entre todos los pares. Es rígido, pero estima un
        parámetro en lugar de cientos de miles, así que casi no tiene ruido. El peso
        δ se calcula, no se elige.</p>
        <p><b>Sin ventas en corto.</b> La mínima varianza sin restricción produjo en
        este proyecto posiciones de hasta −155% del capital: la optimización explota
        los errores de estimación, y si cree por ruido que dos activos se cancelan,
        apalanca esa creencia sin límite. El resultado es matemáticamente óptimo e
        imposible de implementar. Exigir pesos no negativos actúa como regularización
        implícita y la cartera sobrevive mucho mejor fuera de muestra.</p>
    """)


# ==========================================================================
# Volatilidad y pronóstico
# ==========================================================================
def v_volatilidad():
    rets, _, _ = D.cargar()
    lista = sorted(rets.columns)
    ticker = st.selectbox("Activo", lista,
                          index=lista.index("AAPL") if "AAPL" in lista else 0)
    par, vol, pron = D.garch_de(ticker)

    maximo = vol.idxmax()
    E.cabecera(
        "Entregable 3 · Modelo de volatilidad",
        f"¿Cuánto se va a mover {ticker}?",
        f"El modelo estima <b>{pron[0]:.1%}</b> anualizado para mañana y converge a "
        f"<b>{par['vol_largo_plazo']:.1%}</b> en el horizonte largo. Su máximo "
        f"histórico lo sitúa el <b>{maximo:%d de %B de %Y}</b>, sin que ninguna "
        f"variable del modelo identifique ese episodio.",
    )

    E.contexto([
        ("Modelo", "GARCH(1,1)"),
        ("Ajuste", "Variance targeting + rejilla"),
        ("Observaciones", f"{par['n_obs']:,} días"),
        ("Log-verosimilitud", f"{par['log_verosimilitud']:,.0f}"),
    ])

    E.kpis([
        {"clave": "α · reacción", "valor": f"{par['alpha']:.3f}",
         "detalle": "Peso del choque de hoy sobre la varianza de mañana."},
        {"clave": "β · memoria", "valor": f"{par['beta']:.3f}",
         "detalle": "Peso de la varianza de ayer. Es lo que genera las rachas."},
        {"clave": "Persistencia", "termino": "persistencia",
         "valor": f"{par['persistencia']:.3f}",
         "tono": "alto" if par["persistencia"] > .97 else "neutro",
         "detalle": "α + β. Por encima de 0.95 las rachas duran semanas."},
        {"clave": "Vol. de largo plazo", "termino": "volatilidad",
         "valor": f"{par['vol_largo_plazo']:.1%}",
         "detalle": "Nivel al que converge cualquier pronóstico."},
        {"clave": "Agrupamiento", "valor": "SÍ" if par["hay_agrupamiento"] else "NO",
         "tono": "bajo" if par["hay_agrupamiento"] else "alto",
         "detalle": f"Mejora de {par['mejora_vs_constante']:.0f} sobre el modelo de "
                    f"varianza constante. Umbral de significancia: 3."},
    ])

    if not par["hay_agrupamiento"]:
        E.nota(
            "<p>En esta serie el modelo <b>no supera</b> al supuesto de volatilidad "
            "constante, de modo que su persistencia no es interpretable. El "
            "diagnóstico existe precisamente para no reportar un parámetro vacío: "
            "cuando α se aproxima a cero el modelo ignora los choques nuevos y β "
            "deja de estar identificado.</p>",
            "Advertencia de especificación", "alto")

    E.seccion("01", "Volatilidad condicional estimada frente a realizada",
              "La serie estimada usa, para cada día, únicamente información anterior a "
              "ese día. La realizada es la desviación estándar móvil de 21 sesiones y "
              "sirve de contraste.")

    realizada = rets[ticker].rolling(21).std() * np.sqrt(DIAS_HABILES)
    fig, ax = E.lienzo(3.1)
    ax.plot(realizada.index, realizada * 100, color=E.TENUE, linewidth=.75,
            label="realizada · 21 sesiones")
    ax.plot(vol.index, vol * 100, color=E.ALTO, linewidth=1.0, label="GARCH(1,1)")
    ax.axhline(par["vol_largo_plazo"] * 100, color=E.MARCA, linestyle=(0, (4, 3)),
               linewidth=.9, label="largo plazo")
    ax.set_ylabel("volatilidad anualizada (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncols=3)
    E.mostrar(fig)

    E.seccion("02", "Estructura temporal del pronóstico",
              "A partir del segundo día el choque futuro se sustituye por su valor "
              "esperado, y el pronóstico revierte hacia el nivel de largo plazo. La "
              "velocidad de esa reversión la fija la persistencia.")

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        fig, ax = E.lienzo(2.8, 7)
        ax.plot(range(1, len(pron) + 1), pron * 100, color=E.MARCA, linewidth=1.8)
        ax.axhline(par["vol_largo_plazo"] * 100, color=E.SUAVE, linestyle=(0, (4, 3)),
                   linewidth=.9, label="largo plazo")
        ax.set_xlabel("sesiones hacia adelante")
        ax.set_ylabel("volatilidad anualizada (%)")
        ax.legend(loc="center right")
        E.mostrar(fig)
    with der:
        horizonte = pd.DataFrame({
            "h": ["1 sesión", "1 semana", "1 mes", "6 meses", "12 meses"],
            "v": [pron[i] for i in (0, 4, 20, 125, 251)],
        })
        horizonte["d"] = horizonte["v"] - par["vol_largo_plazo"]
        E.tabla(horizonte, {
            "h": {"nombre": "Horizonte"},
            "v": {"nombre": "Pronóstico", "nota": "anualizado", "fmt": pct2, "num": True},
            "d": {"nombre": "vs largo plazo", "nota": "puntos", "fmt": lambda v: f"{v:+.2%}",
                  "num": True},
        }, clases={"d": direccion})
        E.nota(
            "<p>Un promedio móvil no puede producir esta tabla: repetiría el mismo "
            "valor en todos los horizontes.</p>", "", "marca")

    E.detalle("Especificación, ajuste y limitaciones medidas", f"""
        <p><b>Modelo.</b> La varianza condicional sigue la recursión</p>
        <div class="fx">σ²(t) = ω + α · r²(t−1) + β · σ²(t−1)</div>
        <p><b>Ajuste por variance targeting.</b> En lugar de estimar tres parámetros,
        ω se despeja para que la varianza incondicional del modelo coincida con la
        muestral: <code>ω = Var(r) · (1 − α − β)</code>. Quedan dos parámetros libres,
        que se buscan en rejilla con tres refinamientos sucesivos sobre la
        log-verosimilitud gaussiana. Además garantiza que el modelo reproduzca el
        nivel correcto de volatilidad a largo plazo.</p>
        <p><b>Diagnóstico de agrupamiento.</b> Reportar la persistencia sin más puede
        inducir a error: cuando α ≈ 0 el modelo ignora los choques nuevos y β deja de
        estar identificado, de modo que puede tomar cualquier valor sin que el ajuste
        empeore. Sobre ruido blanco, la diferencia de verosimilitud entre β = 0.001 y
        β = 0.99 es de 0.44 en 4,000 observaciones. Por eso el ajuste devuelve también
        la mejora frente al modelo de varianza constante; el umbral 5.99 es el valor
        crítico de una χ² con 2 grados de libertad al 5%.</p>
        <p><b>Limitaciones medidas.</b> Frente al estimador ingenuo —volatilidad de
        las últimas 10 sesiones— el modelo gana en 3 de 5 activos y por margen
        estrecho (0.678 contra 0.625 en XOM). Su ventaja está en el pronóstico
        multiperiodo, que el ingenuo no puede producir. Y subestima los máximos de
        crisis: 91.4% estimado contra 114.9% realizado en marzo de 2020, consecuencia
        de suponer innovaciones normales cuando la distribución real tiene colas más
        pesadas. Se corregiría con una t de Student.</p>
    """)


# ==========================================================================
# Comportamiento en crisis
# ==========================================================================
def v_crisis(n: int):
    tabla_nl, piso = D.no_lineal(n)
    cof = D.cofactores()
    _, _, _, tension = D.mercado_y_garch("v1")

    caidas = tabla_nl["prob_caidas"].mean()
    subidas = tabla_nl["prob_subidas"].mean()
    convergencia = np.corrcoef(cof["beta_calma"], cof["gamma"])[0, 1]

    E.cabecera(
        "Entregable 3 · Comportamiento en crisis",
        "¿Qué pasa con la cobertura cuando el mercado se tensa?",
        f"Se deteriora, por dos vías distintas y medibles. Los activos caen juntos "
        f"<b>más</b> de lo que suben juntos ({caidas:.1%} contra {subidas:.1%}), y la "
        f"beta de la mitad del universo <b>converge hacia 1</b> al entrar en tensión, "
        f"de modo que las posiciones defensivas dejan de serlo.",
    )

    E.contexto([
        ("Pares evaluados", f"{len(tabla_nl):,}"),
        ("Activos con beta condicional", f"{len(cof)}"),
        ("Días en tensión", f"{tension.sum():,} ({tension.mean():.0%})"),
        ("Régimen definido por", "GARCH · quintil superior"),
    ])

    # ---------- bloque 1: dependencia de cola ----------
    E.seccion("01", "Dependencia de cola",
              "Probabilidad de que un activo esté en su decil extremo dado que el otro "
              "lo está. Bajo independencia ambas valdrían 10%. La asimetría entre las "
              "dos es lo que un coeficiente de correlación único no puede expresar, "
              "porque es el mismo número para toda la nube de puntos.")

    E.kpis([
        {"clave": "Caen juntos", "termino": "dependencia de cola",
         "valor": f"{caidas:.1%}", "tono": "alto",
         "detalle": f"{caidas / .10:.1f}× lo que ocurriría por azar."},
        {"clave": "Suben juntos", "termino": "dependencia de cola",
         "valor": f"{subidas:.1%}",
         "detalle": f"{subidas / .10:.1f}× el azar. Menor que a la baja."},
        {"clave": "Asimetría", "valor": f"{caidas - subidas:+.1%}", "tono": "alto",
         "detalle": "Exceso de sincronía a la baja sobre la sincronía al alza."},
        {"clave": "Pares afectados", "valor": f"{(tabla_nl['asimetria'] > 0).mean():.1%}",
         "tono": "alto",
         "detalle": "Proporción con mayor dependencia a la baja. Es sistemático."},
    ])

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        fig, ax = E.lienzo(2.9, 7)
        ax.hist(tabla_nl["prob_caidas"], bins=38, alpha=.72, color=E.ALTO,
                edgecolor="none", label=f"caídas simultáneas · media {caidas:.1%}")
        ax.hist(tabla_nl["prob_subidas"], bins=38, alpha=.62, color=E.BAJO,
                edgecolor="none", label=f"alzas simultáneas · media {subidas:.1%}")
        ax.axvline(.10, color=E.TINTA, linestyle=(0, (4, 3)), linewidth=1.1,
                   label="independencia · 10%")
        ax.set_xlabel("P(el otro activo también está en su decil extremo)")
        ax.set_ylabel("pares")
        ax.legend(loc="upper right")
        E.mostrar(fig)
    with der:
        E.html('<div class="sec-sub" style="margin-top:4px">Pares con mayor '
                    'asimetría: los que peor se comportan como cobertura.</div>',
                    )
        E.tabla(
            tabla_nl.nlargest(9, "asimetria"),
            {
                "activo_a": {"nombre": "A", "tick": True},
                "activo_b": {"nombre": "B", "tick": True},
                "prob_caidas": {"nombre": "Caen", "nota": "juntos", "fmt": pct1, "num": True},
                "prob_subidas": {"nombre": "Suben", "nota": "juntos", "fmt": pct1, "num": True},
            },
        )

    # ---------- bloque 2: no linealidad ----------
    supera = (tabla_nl["brecha"] > piso).mean()
    razon = supera / .05

    E.seccion("02", "Dependencia no lineal",
              "La correlación de Pearson es la pendiente de la mejor recta y no detecta "
              "relaciones curvas. La información mutua sí: mide cualquier forma de "
              "dependencia y vale cero solo bajo independencia.")

    E.kpis([
        {"clave": "Brecha media", "valor": f"{tabla_nl['brecha'].mean():+.4f}",
         "detalle": "Dependencia detectada por encima de la que captura Pearson."},
        {"clave": "Piso del método", "valor": f"{piso:.4f}",
         "detalle": "Brecha que aparece entre series sin relación alguna."},
        {"clave": "Pares sobre el piso", "valor": f"{supera:.1%}",
         "tono": "aviso" if razon < 3 else "alto",
         "detalle": "Por construcción, el 5% lo supera por azar."},
        {"clave": "Razón sobre el azar", "valor": f"{razon:.1f}", "unidad": "×",
         "tono": "aviso" if razon < 3 else "alto",
         "detalle": "Por debajo de 3 no constituye evidencia sólida."},
    ])

    if razon < 3:
        E.nota(
            f"<p><b>Resultado negativo, y se reporta como tal.</b> El {supera:.1%} de "
            f"los pares supera el piso del método, apenas {razon:.1f} veces el 5% que "
            f"lo superaría por construcción. La información mutua no encuentra "
            f"evidencia sólida de relaciones curvas en rendimientos diarios.</p>"
            f"<p>La conclusión no es que no haya no linealidad, sino que <b>no está en "
            f"la forma de la relación entre pares</b>. Está en las colas, medida en el "
            f"bloque anterior, y en el régimen, medido en el siguiente.</p>",
            "Hallazgo", "aviso")

    E.detalle("Cómo se calibra la información mutua", """
        <p><b>Estimación.</b> Las dos series se discretizan en 8 casillas por cuantiles,
        no por intervalos de igual ancho: con colas pesadas, los intervalos uniformes
        dejarían casi todas las observaciones en la casilla central y las colas vacías,
        que es justo donde está la información. Después se compara la tabla conjunta
        contra el producto de las marginales.</p>
        <p><b>Escala comparable.</b> La información mutua se mide en nats y no tiene
        tope, así que se traduce invirtiendo la identidad válida para una normal
        bivariada:</p>
        <div class="fx">MI = −½ · log(1 − ρ²)   ⟹   ρ equivalente = √(1 − e^(−2·MI))

brecha = ρ equivalente − |ρ de Pearson|</div>
        <p><b>Corrección de sesgo.</b> La información mutua estimada de una muestra
        finita nunca sale cero, ni con datos independientes. Se resta el valor esperado
        bajo independencia, <code>(bins−1)²/(2n)</code>. La corrección se validó
        permutando una serie 300 veces: sesgo empírico 0.00842 ± 0.00175 contra
        analítico 0.00835.</p>
        <p><b>Piso de resolución.</b> Discretizar <i>pierde</i> información y el sesgo
        residual la <i>agrega</i>, y no se puede saber a priori cuál domina. Por eso se
        mide: se permutan pares reales, se observa qué brecha aparece entre series sin
        relación, y el percentil 95 de esa distribución es el umbral. Sin esta
        calibración, una brecha de 0.03 no sería interpretable.</p>
    """)

    # ---------- bloque 3: cofactores ----------
    E.seccion("03", "Beta condicional al régimen",
              "Modelo con término de interacción. El coeficiente γ mide cuánto cambia la "
              "sensibilidad al mercado al pasar de calma a tensión; el régimen no se "
              "elige a mano, lo define la volatilidad condicional del GARCH.")

    E.html('<div class="fx">r_activo = α + β · r_mercado + γ · (r_mercado × tensión) + ε\n'
                '\n'
                '  calma   (tensión = 0)  →  sensibilidad = β\n'
                '  tensión (tensión = 1)  →  sensibilidad = β + γ</div>',
                )

    E.kpis([
        {"clave": "Con γ significativa", "termino": "significativo",
         "valor": f"{cof['significativo'].mean():.1%}", "tono": "alto",
         "detalle": f"{int(cof['significativo'].sum())} de {len(cof)} activos, al 5% "
                    f"con errores estándar robustos."},
        {"clave": "corr(β calma, γ)", "valor": f"{convergencia:+.3f}", "tono": "alto",
         "detalle": "Negativa: las betas bajas suben y las altas bajan."},
        {"clave": "Dispersión de β", "valor": f"{cof['beta_calma'].std():.3f} → "
                                              f"{cof['beta_tension'].std():.3f}",
         "detalle": f"Se comprime {1 - cof['beta_tension'].std() / cof['beta_calma'].std():.0%} "
                    f"al entrar en tensión."},
        {"clave": "Distancia media a 1", "valor": f"{(cof['beta_calma'] - 1).abs().mean():.3f} → "
                                                  f"{(cof['beta_tension'] - 1).abs().mean():.3f}",
         "detalle": "Las betas se acercan a la del mercado."},
    ])

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        fig, ax = E.lienzo(4.3, 6.4)
        colores = np.where(cof["significativo"], E.ALTO, E.TENUE)
        ax.scatter(cof["beta_calma"], cof["beta_tension"], s=13, c=colores,
                   alpha=.6, edgecolors="none")
        lim = [cof[["beta_calma", "beta_tension"]].min().min() - .08,
               cof[["beta_calma", "beta_tension"]].max().max() + .08]
        ax.plot(lim, lim, color=E.TINTA, linewidth=.9, linestyle=(0, (4, 3)),
                label="sin cambio de régimen")
        ax.axhline(1, color=E.MARCA, linewidth=.8, alpha=.5)
        ax.axvline(1, color=E.MARCA, linewidth=.8, alpha=.5)
        ax.set_xlabel("beta en calma")
        ax.set_ylabel("beta en tensión")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(axis="x")
        ax.legend(loc="upper left")
        E.mostrar(fig)
        E.html('<div class="sec-sub">Cada punto es un activo. En rojo los que '
                    'tienen γ significativa. Si la beta no cambiara con el régimen, '
                    'todos caerían sobre la diagonal; en cambio giran alrededor de 1.</div>',
                    )
    with der:
        q = cof.copy()
        q["quintil"] = pd.qcut(q["beta_calma"], 5, labels=[1, 2, 3, 4, 5])
        agg = q.groupby("quintil", observed=True).agg(
            n=("ticker", "size"), calma=("beta_calma", "mean"),
            tension=("beta_tension", "mean"), gamma=("gamma", "mean"),
        ).reset_index()
        agg["quintil"] = agg["quintil"].astype(str)
        E.tabla(agg, {
            "quintil": {"nombre": "Q", "nota": "β en calma", "termino": "quintil"},
            "n": {"nombre": "n", "fmt": ent, "num": True},
            "calma": {"nombre": "β calma", "fmt": num3, "num": True, "termino": "beta"},
            "tension": {"nombre": "β tensión", "fmt": num3, "num": True,
                        "termino": "beta condicional"},
            "gamma": {"nombre": "γ", "nota": "cambio", "fmt": sgn3, "num": True},
        }, clases={"gamma": direccion})
        E.nota(
            "<p>γ es monótona a lo largo de los quintiles: positiva en las defensivas, "
            "negativa en las agresivas. Eso es convergencia, no ruido.</p>",
            "", "marca")

    E.seccion("04", "Efecto por sector",
              "Qué sectores pierden capacidad defensiva al entrar en tensión y cuáles "
              "la ganan.")

    sec = cof.groupby("sector").agg(
        n=("ticker", "size"), calma=("beta_calma", "mean"),
        tension=("beta_tension", "mean"), gamma=("gamma", "mean"),
        sig=("significativo", "mean"),
    ).sort_values("gamma", ascending=False).reset_index()

    izq, der = st.columns([2, 3], gap="medium")
    with izq:
        fig, ax = E.lienzo(4.0, 5.4)
        E.barras(ax, sec["sector"], sec["gamma"],
                 colores=np.where(sec["gamma"] > 0, E.ALTO, E.BAJO), fmt="{:+.3f}")
        ax.set_xlabel("γ promedio · cambio de beta al tensarse el mercado")
        E.mostrar(fig)
    with der:
        E.tabla(sec, {
            "sector": {"nombre": "Sector"},
            "n": {"nombre": "n", "fmt": ent, "num": True},
            "calma": {"nombre": "β calma", "fmt": num3, "num": True, "termino": "beta"},
            "tension": {"nombre": "β tensión", "fmt": num3, "num": True,
                        "termino": "beta condicional"},
            "gamma": {"nombre": "γ", "fmt": sgn3, "num": True},
            "sig": {"nombre": "Significativas", "nota": "% del sector", "fmt": pct0,
                    "num": True, "termino": "significativo"},
        }, clases={"gamma": direccion},
           destaca=lambda f: f["sector"] == "Utilities")

    E.nota(
        "<p><b>Utilities es el sector más defensivo del índice y el que más capacidad "
        "defensiva pierde</b>: su beta pasa de 0.479 a 0.779, con el 93% de sus "
        "componentes significativos. La posición que se toma para protegerse deja de "
        "proteger exactamente en el escenario para el que se tomó.</p>"
        "<p>Information Technology hace lo contrario: es el más agresivo en calma "
        "(1.291) y se modera en tensión (1.105). Las dos puntas se mueven hacia el "
        "centro.</p>",
        "Lectura", "alto")

    E.seccion("05", "Forma del efecto",
              "Versión no paramétrica del modelo: la beta se estima por separado dentro "
              "de cada nivel de volatilidad, sin imponer que el cambio sea un salto "
              "único ni que sea lineal en el nivel.")

    c1, c2 = st.columns([1, 3], gap="medium")
    elegido = c1.selectbox("Activo", cof.sort_values("gamma", ascending=False)["ticker"])
    grupos = c1.slider("Niveles de volatilidad", 3, 8, 5)
    esc = D.escalones(elegido, grupos)

    with c2:
        fig, ax = E.lienzo(2.9, 7)
        ax.errorbar(esc["nivel"], esc["beta"], yerr=1.96 * esc["ee_robusto"],
                    fmt="o-", color=E.MARCA, capsize=3.5, linewidth=1.5,
                    markersize=5, elinewidth=1)
        ax.set_xlabel("nivel de volatilidad del mercado  ·  1 = más tranquilo")
        ax.set_ylabel(f"beta de {elegido}")
        ax.set_xticks(esc["nivel"])
        E.mostrar(fig)

    E.tabla(esc, {
        "nivel": {"nombre": "Nivel", "nota": "1 = calma", "fmt": ent, "num": True},
        "dias": {"nombre": "Sesiones", "fmt": ent, "num": True},
        "vol_mercado": {"nombre": "Vol. del mercado", "nota": "anualizada",
                        "fmt": pct1, "num": True, "termino": "volatilidad"},
        "beta": {"nombre": "Beta", "fmt": num3, "num": True, "termino": "beta"},
        "ee_robusto": {"nombre": "Error estándar", "nota": "robusto", "fmt": num3,
                       "num": True, "termino": "error estandar robusto"},
        "r2": {"nombre": "R²", "nota": "explicado por el mercado", "fmt": num3,
               "num": True, "termino": "R cuadrado"},
    })

    E.nota(
        "<p>Las barras verticales son intervalos de confianza al 95%. Cuando se "
        "traslapan, la diferencia entre niveles <b>no es distinguible del azar</b> y no "
        "debe interpretarse. Lo informativo es si los niveles superiores se despegan "
        "del resto, y la columna R²: mide qué fracción del movimiento del activo "
        "explica el mercado, y suele multiplicarse por varias veces en tensión.</p>",
        "Cómo leer el gráfico", "marca")

    E.detalle("Por qué errores estándar robustos, y no los clásicos", """
        <p>Los errores estándar de mínimos cuadrados convencionales suponen que la
        varianza del error es constante. Este mismo proyecto demostró que no lo es: el
        modelo GARCH estimó persistencias de 0.91 a 0.99 sobre los activos analizados.
        Usar errores estándar que suponen lo contrario contradiría el hallazgo previo,
        y lo hace en la dirección peligrosa: los subestima, lo que infla los
        estadísticos t y hace parecer significativo lo que no lo es.</p>
        <p>Se emplean por tanto los errores estándar robustos a heterocedasticidad de
        White, variante HC1 con corrección por grados de libertad:</p>
        <div class="fx">V = (X'X)⁻¹ · X' diag(e²) X · (X'X)⁻¹ · n/(n−k)</div>
        <p>En lugar de una varianza única para todos los días, cada día aporta su
        propio residuo. Medido sobre los 457 activos, <b>los errores estándar clásicos
        subestiman los robustos en un factor de 1.54</b>.</p>
        <p><b>Una identidad, no un resultado.</b> La γ promedio del universo es
        exactamente cero y las betas promedian exactamente 1 en ambos regímenes. Eso no
        es un hallazgo: el mercado se define como el promedio equiponderado de estos
        mismos activos, de modo que la beta promedio contra él vale 1 en cualquier
        submuestra. La consecuencia práctica importa: <b>el agregado no puede revelar
        nada</b> sobre el efecto del régimen, y toda la información está en la
        dispersión.</p>
    """)


# ==========================================================================
# Pruebas de estres
# ==========================================================================
def v_estres(cartera: dict):
    sel, metodo, encoger = cartera["posiciones"], cartera["metodo"], cartera["encoger"]
    if len(sel) < 2:
        E.cabecera("Entregable 4 · Pruebas de estrés",
                   "¿Qué le pasa a esta cartera si se repite una crisis?",
                   "Define una cartera de al menos dos posiciones en la barra lateral.")
        return

    # El deslizador trabaja en puntos porcentuales enteros para que su etiqueta
    # se lea "-10%" y no "-0%", que es lo que produce formatear un -0.10 con
    # un formato entero. El valor se convierte a fraccion aqui.
    choque = st.session_state.get("choque_pct", -10) / 100
    r = D.estres(tuple(sel), metodo, encoger, choque)
    v95, v99, caida = r["var95"], r["var99"], r["caida"]

    E.cabecera(
        "Entregable 4 · Pruebas de estrés",
        "¿Qué le pasa a esta cartera si se repite una crisis?",
        f"En el peor episodio del periodo habría perdido <b>{caida['caida']:.1%}</b> "
        f"en {caida['dias_de_caida']} sesiones. Su volatilidad se multiplica por "
        f"<b>{r['regimenes'].set_index('regimen').loc['Tensión', 'vol_portafolio'] / r['regimenes'].set_index('regimen').loc['Calma', 'vol_portafolio']:.1f}</b> "
        f"al pasar de calma a tensión, y un modelo de beta constante subestimaría "
        f"la pérdida ante un choque en <b>{r['choque']['subestimacion_relativa']:.1%}</b>.",
    )

    E.contexto([
        ("Posiciones", str(len(sel))),
        ("Asignación", metodo),
        ("Historia", f"{len(r['serie']):,} sesiones"),
        ("Medidas", "VaR y ES históricos"),
    ])

    E.kpis([
        {"clave": "VaR 95% · 1 día", "valor": f"{v95['var']:.2%}", "tono": "aviso",
         "detalle": "Pérdida mínima esperada en el peor 5% de las sesiones."},
        {"clave": "Expected Shortfall 95%", "valor": f"{v95['expected_shortfall']:.2%}",
         "tono": "alto",
         "detalle": f"Pérdida media de esa cola ({v95['dias_en_la_cola']} sesiones). "
                    f"Es {v95['razon_es_var']:.2f}× el VaR."},
        {"clave": "VaR 99% · 1 día", "valor": f"{v99['var']:.2%}", "tono": "aviso",
         "detalle": f"Expected Shortfall al 99%: {v99['expected_shortfall']:.2%}."},
        {"clave": "Máxima caída", "valor": f"{caida['caida']:.1%}", "tono": "alto",
         "detalle": f"{caida['pico']:%d %b %Y} → {caida['fondo']:%d %b %Y}. "
                    + (f"Recuperada en {caida['dias_de_recuperacion']} sesiones."
                       if caida["recuperacion"] is not None else "Sin recuperar.")},
    ])

    E.detalle("Por qué VaR histórico y por qué acompañado de Expected Shortfall", """
        <p><b>Histórico y no paramétrico.</b> Ambas medidas se leen directamente de la
        distribución observada, sin suponer normalidad. Es deliberado: este mismo
        proyecto documentó que los rendimientos tienen colas más pesadas que la normal
        —es la razón por la que el GARCH subestima los máximos de crisis— de modo que
        un VaR paramétrico gaussiano heredaría ese error justo en la cola que pretende
        medir.</p>
        <p><b>El Expected Shortfall acompaña al VaR y no al revés.</b> El VaR dice
        cuánto se pierde en el peor 5% de las sesiones <i>en el mejor de esos casos</i>;
        no dice nada sobre qué tan grave puede ser el resto de la cola. El Expected
        Shortfall promedia toda esa cola, y por eso es la medida que el marco de
        Basilea adoptó en sustitución del VaR. La razón ES/VaR de esta cartera es
        """ + f"{v95['razon_es_var']:.2f}" + """: cuanto más alta, más pesada la cola.</p>
    """)

    # ---------- escenarios historicos ----------
    E.seccion("01", "Reproducción de episodios reales",
              "La cartera actual sometida a los rendimientos que efectivamente "
              "ocurrieron en cada ventana. No hay modelo de por medio ni parámetros "
              "que calibrar: es aritmética sobre datos observados.")

    esc = r["escenarios"]
    if len(esc):
        izq, der = st.columns([3, 2], gap="medium")
        with izq:
            E.tabla(esc, {
                "escenario": {"nombre": "Episodio"},
                "sesiones": {"nombre": "Sesiones", "fmt": ent, "num": True},
                "rendimiento": {"nombre": "Resultado", "nota": "acumulado",
                                "fmt": pct1, "num": True},
                "caida_maxima": {"nombre": "Máxima caída", "nota": "pico a valle",
                                 "fmt": pct1, "num": True},
                "peor_dia": {"nombre": "Peor sesión", "fmt": pct1, "num": True},
                "volatilidad": {"nombre": "Volatilidad", "nota": "anualizada",
                                "fmt": pct1, "num": True, "termino": "volatilidad"},
            }, clases={"rendimiento": direccion, "caida_maxima": direccion,
                       "peor_dia": direccion})
        with der:
            orden = esc.sort_values("rendimiento")
            fig, ax = E.lienzo(2.9, 5.6)
            E.barras(ax, orden["escenario"], orden["rendimiento"] * 100,
                     colores=np.where(orden["rendimiento"] < 0, E.ALTO, E.BAJO),
                     fmt="{:+.1f}%")
            ax.set_xlabel("resultado acumulado (%)")
            E.mostrar(fig)

    E.seccion("02", "Peores ventanas detectadas en los datos",
              "Complementa a los episodios con nombre, que los eligió una persona. "
              "Estas salen del propio panel: si el peor tramo del periodo no coincide "
              "con ninguno de los nombrados, aparece aquí de todos modos. Se exige "
              "separación de 90 días para que no sean cinco recortes del mismo "
              "desplome.")
    E.tabla(r["ventanas"], {
        "inicio": {"nombre": "Inicio", "fmt": lambda d: f"{d:%Y-%m-%d}"},
        "fin": {"nombre": "Fin", "fmt": lambda d: f"{d:%Y-%m-%d}"},
        "rendimiento": {"nombre": "Resultado", "nota": "21 sesiones",
                        "fmt": pct1, "num": True},
    }, clases={"rendimiento": direccion})

    # ---------- regimenes ----------
    reg = r["regimenes"].set_index("regimen")
    E.seccion("03", "Riesgo condicional al régimen",
              "La misma cartera valuada con la covarianza de los días de calma y con "
              "la de los días de tensión, en lugar de una sola para toda la historia. "
              "La diferencia mide cuánta protección desaparece justo cuando hace falta.")

    E.kpis([
        {"clave": "Volatilidad en calma", "termino": "volatilidad",
         "valor": f"{reg.loc['Calma', 'vol_portafolio']:.1%}", "tono": "bajo",
         "detalle": f"{int(reg.loc['Calma', 'sesiones']):,} sesiones."},
        {"clave": "Volatilidad en tensión", "termino": "volatilidad",
         "valor": f"{reg.loc['Tensión', 'vol_portafolio']:.1%}", "tono": "alto",
         "detalle": f"{reg.loc['Tensión', 'vs_completo']:+.0%} frente al periodo completo."},
        {"clave": "Diversificación en calma", "termino": "ratio de diversificacion",
         "valor": f"{reg.loc['Calma', 'ratio_diversificacion']:.3f}", "tono": "bajo",
         "detalle": "Cuánto riesgo se cancela entre posiciones."},
        {"clave": "Diversificación en tensión", "termino": "ratio de diversificacion",
         "valor": f"{reg.loc['Tensión', 'ratio_diversificacion']:.3f}", "tono": "alto",
         "detalle": f"Pierde "
                    f"{1 - reg.loc['Tensión', 'ratio_diversificacion'] / reg.loc['Calma', 'ratio_diversificacion']:.0%} "
                    f"de su capacidad de cancelación."},
    ])

    E.tabla(r["regimenes"], {
        "regimen": {"nombre": "Régimen"},
        "sesiones": {"nombre": "Sesiones", "fmt": ent, "num": True},
        "vol_portafolio": {"nombre": "Vol. de la cartera", "nota": "anualizada",
                           "fmt": pct2, "num": True},
        "vol_ponderada": {"nombre": "Suma ponderada", "nota": "sin cancelación",
                          "fmt": pct2, "num": True},
        "ratio_diversificacion": {"nombre": "Diversificación", "fmt": num3, "num": True,
                                  "termino": "ratio de diversificacion"},
        "vs_completo": {"nombre": "vs periodo completo", "fmt": lambda v: f"{v:+.1%}",
                        "num": True},
    }, clases={"vs_completo": direccion},
       destaca=lambda f: f["regimen"] == "Tensión")

    # ---------- choque ----------
    c = r["choque"]
    E.seccion("04", "Proyección ante un choque de mercado",
              "Qué pérdida proyecta cada uno de los dos modelos que el proyecto "
              "estimó. La diferencia entre ambos es, en puntos de rendimiento, el "
              "costo de suponer que la beta no cambia con el régimen.")

    st.slider("Magnitud del choque de mercado", min_value=-40, max_value=-1,
              value=int(round(choque * 100)), step=1, key="choque_pct",
              format="%d%%",
              help="Movimiento del índice sobre el que se proyecta la pérdida. "
                   "Marzo de 2020 fue del orden de -34% en un mes.")

    E.kpis([
        {"clave": "Modelo de beta constante", "termino": "beta",
         "valor": f"{c['perdida_modelo_estandar']:.2%}", "tono": "aviso",
         "detalle": f"Beta única de la cartera: {c['beta_unica']:.3f}."},
        {"clave": "Modelo de beta condicional", "termino": "beta condicional",
         "valor": f"{c['perdida_modelo_condicional']:.2%}", "tono": "alto",
         "detalle": f"Beta en tensión: {c['beta_tension']:.3f}."},
        {"clave": "Error del modelo estándar",
         "valor": f"{c['error_del_modelo_estandar']:.2%}", "tono": "alto",
         "detalle": "Pérdida adicional que el modelo estándar no anticipa."},
        {"clave": "Subestimación", "valor": f"{c['subestimacion_relativa']:.1%}",
         "tono": "alto",
         "detalle": "En términos relativos sobre la pérdida proyectada."},
    ])

    E.nota(
        f"<p>La beta de esta cartera pasa de <b>{c['beta_calma']:.3f}</b> en calma a "
        f"<b>{c['beta_tension']:.3f}</b> en tensión. La beta única del modelo estándar "
        f"vale <b>{c['beta_unica']:.3f}</b> y cae entre las dos, lo cual no es "
        f"casualidad: se estima sobre todas las sesiones, de las que una quinta parte "
        f"son de tensión, así que <b>ya absorbe parte del salto</b>.</p>"
        f"<p>Por eso el error real de usarla ({c['salto_vs_beta_unica']:+.4f}) es menor "
        f"que el salto completo γ de la cartera ({c['gamma_cartera']:+.4f}). Reportar γ "
        f"como si fuera el error exageraría el problema, y este proyecto no necesita "
        f"exagerarlo.</p>",
        "γ y el error del modelo son cantidades distintas", "marca")

    E.seccion("05", "Desglose del error por posición",
              "Qué posiciones concentran la pérdida que el modelo estándar no anticipa. "
              "No son las más volátiles: son las que combinan peso alto con un salto "
              "grande de beta.")

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        E.tabla(r["detalle"], {
            "ticker": {"nombre": "Activo", "tick": True},
            "sector": {"nombre": "Sector"},
            "peso": {"nombre": "Peso", "fmt": pct1, "num": True},
            "beta_unica": {"nombre": "β única", "fmt": num3, "num": True,
                           "termino": "beta"},
            "beta_tension": {"nombre": "β tensión", "fmt": num3, "num": True,
                             "termino": "beta condicional"},
            "aporte_condicional": {"nombre": "Aporte a la pérdida", "fmt": pct2,
                                   "num": True},
            "error": {"nombre": "No anticipado", "nota": "por el modelo estándar",
                      "fmt": pct2, "num": True},
        }, clases={"error": direccion})
    with der:
        d = r["detalle"]
        fig, ax = E.lienzo(max(2.4, .33 * len(d) + 1.0), 5.4)
        E.barras(ax, d["ticker"], d["error"] * 100,
                 colores=np.where(d["error"] < 0, E.ALTO, E.BAJO), fmt="{:+.2f}%")
        ax.set_xlabel("pérdida no anticipada (%)")
        E.mostrar(fig)

    peor = r["detalle"].iloc[0]
    if metodo == D.MINIMA_VARIANZA:
        E.nota(
            f"<p>La posición que más error concentra es <b>{peor['ticker']}</b> "
            f"({peor['sector']}), con {peor['peso']:.1%} del capital y una beta que "
            f"pasa de {peor['beta_unica']:.3f} a {peor['beta_tension']:.3f}.</p>"
            f"<p><b>No es coincidencia.</b> La optimización de mínima varianza se "
            f"concentra en los activos de menor volatilidad, que son los defensivos; "
            f"y los defensivos son precisamente los que más capacidad defensiva "
            f"pierden al entrar en tensión, como muestra la vista de comportamiento "
            f"en crisis. La cartera que parece más segura es la que peor describe el "
            f"modelo estándar. Cambia la asignación a equiponderada en la barra "
            f"lateral para comparar.</p>",
            "El optimizador se concentra justo donde falla el supuesto", "alto")

    E.detalle("Alcance y límites de estas pruebas", """
        <p><b>Lo que sí sostiene.</b> La reproducción de episodios reales no supone
        nada: son los rendimientos que ocurrieron aplicados a la cartera de hoy. El
        riesgo por régimen y la proyección del choque sí usan modelos, pero sus
        parámetros están estimados sobre el mismo panel y documentados en sus vistas.</p>
        <p><b>Lo que no sostiene.</b> El pasado no agota el espacio de lo posible: la
        peor caída de la muestra no es la peor caída concebible, y una crisis con una
        estructura distinta —un choque sectorial en lugar de uno de mercado— no está
        representada. La proyección del choque supone además que la relación con el
        mercado se mantiene lineal dentro de cada régimen, cosa que en los movimientos
        más extremos tiende a romperse.</p>
        <p><b>Lo que haría falta para ir más lejos.</b> Escenarios por factor en lugar
        de un único factor de mercado, y una distribución con colas pesadas —una t de
        Student— para generar escenarios sintéticos más severos que los observados.</p>
    """)


# ==========================================================================
# Validación
# ==========================================================================
def v_validacion(n: int):
    tabla = D.barrido(n)
    mejor = tabla.loc[tabla["vol_realizada"].idxmin()]

    E.cabecera(
        "Entregable 3 · Validación del modelo",
        "¿Cuánta historia conviene usar para estimar el riesgo?",
        f"<b>{int(mejor['ventana_dias'])} sesiones</b>, aproximadamente un año. "
        f"Ventanas más largas producen carteras <b>más</b> volátiles, no menos: los "
        f"mercados cambian y los datos antiguos describen un régimen que ya no "
        f"aplica.",
    )

    E.contexto([
        ("Método", "Validación hacia adelante"),
        ("Rebalanceo", "21 sesiones"),
        ("Rebalanceos", f"{int(mejor['n_rebalanceos'])}"),
        ("Inicio común", f"sesión {int(mejor['evalua_desde_dia'])}"),
        ("Muestra", f"{n} activos"),
    ])

    E.kpis([
        {"clave": "Ventana óptima", "valor": f"{int(mejor['ventana_dias'])}", "unidad": "d",
         "tono": "marca", "detalle": f"{mejor['ventana_anios']:.1f} años de historia."},
        {"clave": "Volatilidad realizada", "termino": "volatilidad",
         "valor": f"{mejor['vol_realizada']:.2%}",
         "detalle": "La que efectivamente tuvo la cartera fuera de muestra."},
        {"clave": "Rotación media", "termino": "rotacion",
         "valor": f"{mejor['rotacion_media']:.1%}",
         "detalle": "Proporción de la cartera que cambia en cada rebalanceo."},
        {"clave": "Penalización por exceso", "valor":
            f"{tabla['vol_realizada'].max() / mejor['vol_realizada'] - 1:+.1%}",
         "tono": "alto",
         "detalle": "Volatilidad adicional de la peor ventana frente a la óptima."},
    ])

    izq, der = st.columns([3, 2], gap="medium")
    with izq:
        fig, ax = E.lienzo(3.0, 7)
        ax.plot(tabla["ventana_dias"], tabla["vol_realizada"] * 100, "o-",
                color=E.MARCA, linewidth=1.6, markersize=5)
        ax.scatter([mejor["ventana_dias"]], [mejor["vol_realizada"] * 100], s=150,
                   facecolors="none", edgecolors=E.ALTO, linewidths=1.8, zorder=5,
                   label="óptimo")
        ax.set_xlabel("sesiones de historia utilizadas")
        ax.set_ylabel("volatilidad realizada (%)")
        ax.legend(loc="upper left")
        E.mostrar(fig)
    with der:
        E.tabla(tabla, {
            "ventana_dias": {"nombre": "Ventana", "nota": "sesiones", "fmt": ent, "num": True},
            "ventana_anios": {"nombre": "Años", "fmt": lambda v: f"{v:.1f}", "num": True},
            "vol_realizada": {"nombre": "Vol. realizada", "nota": "anualizada",
                              "fmt": pct2, "num": True},
            "rotacion_media": {"nombre": "Rotación", "nota": "por rebalanceo",
                               "fmt": pct1, "num": True, "termino": "rotacion"},
        }, destaca=lambda f: f["ventana_dias"] == mejor["ventana_dias"])

    E.nota(
        "<p>La primera versión de esta prueba concluía lo contrario —2,000 sesiones— "
        "porque las ventanas <b>medían periodos distintos</b>: la de 250 sesiones "
        "alcanzaba a atravesar el episodio de marzo de 2020 y acumulaba 133 "
        "rebalanceos, mientras que la de 2,000 comenzaba después y solo tenía 44. No "
        "eran comparables.</p>"
        "<p>Al fijar un inicio de evaluación común a todas, la conclusión se invirtió. "
        "Una prueba posterior detectó además que una ventana mayor que la muestra "
        "contaminaba el cálculo de ese inicio común.</p>",
        "Error detectado y corregido", "aviso")

    E.detalle("Procedimiento y control de look-ahead", """
        <p>La prueba del Entregable 2 partía la historia una sola vez, 70% para estimar
        y 30% para medir. Eso tiene dos problemas: el resultado depende de qué le tocó
        al 30% final, y no dice nada sobre cuánta historia conviene usar.</p>
        <p>La validación hacia adelante repite el ejercicio avanzando en el tiempo:</p>
        <div class="fx">ventana de W sesiones → construye cartera → la mantiene 21 sesiones → mide
avanza 21 sesiones y repite</div>
        <p>En cada paso el modelo <b>solo observa el pasado</b>: nunca utiliza
        información del periodo que está prediciendo. Por eso el resultado es una
        estimación honesta del comportamiento en producción y no un ajuste dentro de
        muestra.</p>
        <p><b>Restricción que define el tamaño del universo.</b> Con N activos, la
        covarianza muestral necesita más de N observaciones para ser invertible: con
        150 activos, una ventana de 125 sesiones ni siquiera puede usarse. El
        encogimiento de Ledoit-Wolf sí lo permite, porque encoge hacia una matriz que
        siempre es invertible, y eso en sí mismo es un resultado que vale medir.</p>
    """)


# ==========================================================================
# Metodología
# ==========================================================================
def v_metodo():
    E.cabecera(
        "Documentación",
        "Metodología, límites y glosario",
        "Las decisiones que condicionan los resultados, las limitaciones medidas y "
        "los errores detectados durante el desarrollo. Cada limitación está "
        "cuantificada, no estimada.",
    )

    E.seccion("01", "Decisiones metodológicas")
    E.tabla(pd.DataFrame([
        {"d": "Rendimientos logarítmicos, no precios",
         "p": "Dos series con tendencia compartida darían correlación alta sin relación económica.",
         "e": "Evita correlación espuria"},
        {"d": "Precios ajustados por splits y dividendos",
         "p": "Un split 4:1 aparecería como una caída de −75% en una sesión.",
         "e": "Evita falsos eventos extremos"},
        {"d": "Panel balanceado, sin imputación",
         "p": "Rellenar un precio ausente inventa una sesión de rendimiento cero.",
         "e": "Menos activos, datos reales"},
        {"d": "Encogimiento de Ledoit-Wolf",
         "p": "N(N+1)/2 parámetros contra T observaciones: la matriz muestral queda mal condicionada.",
         "e": "Carteras estables fuera de muestra"},
        {"d": "Mínima varianza sin ventas en corto",
         "p": "La versión libre alcanzó posiciones de −155%: explota errores de estimación.",
         "e": "Regularización implícita"},
        {"d": "Errores estándar robustos de White",
         "p": "El propio GARCH demostró que la varianza no es constante.",
         "e": "Significancia no inflada"},
        {"d": "Régimen definido por el modelo",
         "p": "Elegir el corte a criterio permitiría seleccionar el resultado conveniente.",
         "e": "Sin grados de libertad ocultos"},
        {"d": "Mercado como promedio equiponderado",
         "p": "Sin capitalizaciones diarias, es la aproximación honesta al primer componente principal.",
         "e": "Beta promedio = 1 por construcción"},
    ]), {
        "d": {"nombre": "Decisión"},
        "p": {"nombre": "Motivo"},
        "e": {"nombre": "Efecto"},
    })

    E.seccion("02", "Limitaciones", "Todas cuantificadas sobre los datos del proyecto.")
    E.tabla(pd.DataFrame([
        {"l": "El GARCH subestima los máximos de crisis",
         "m": "91.4% estimado contra 114.9% realizado · marzo 2020",
         "c": "Supone innovaciones normales; la distribución real tiene colas pesadas",
         "s": "Distribución t de Student"},
        {"l": "Ventaja estrecha sobre el estimador ingenuo a un día",
         "m": "Gana en 3 de 5 activos · 0.678 contra 0.625 en XOM",
         "c": "Resultado documentado de la literatura a horizonte corto",
         "s": "Ninguna: su ventaja es el pronóstico multiperiodo"},
        {"l": "La información mutua no detecta no linealidad",
         "m": "6.0% de los pares supera el piso, contra 5% por azar",
         "c": "Discretizar pierde información y el sesgo residual la agrega",
         "s": "Ninguna: el resultado negativo es el hallazgo"},
        {"l": "El umbral de régimen usa la muestra completa",
         "m": "0.25% de las sesiones cambia de lado al recalcular",
         "c": "El cuantil conoce toda la historia; la volatilidad condicional no",
         "s": "Variante estrictamente causal disponible"},
        {"l": "Índice equiponderado en lugar de ponderado por capitalización",
         "m": "La beta promedio del universo vale 1 por construcción",
         "c": "No se dispone de capitalizaciones diarias históricas",
         "s": "Ponderar por capitalización histórica"},
        {"l": "Los escenarios de estrés no exceden lo observado",
         "m": "La peor ventana del panel es −33.5% (marzo 2020)",
         "c": "Reproducen episodios reales; el pasado no agota lo posible",
         "s": "Escenarios sintéticos con distribución de colas pesadas"},
        {"l": "La proyección del choque supone linealidad dentro del régimen",
         "m": "La subestimación relativa es constante en el tamaño del choque",
         "c": "El modelo con cofactor es lineal en el rendimiento del mercado",
         "s": "Término cuadrático, o escenarios por factor sectorial"},
    ]), {
        "l": {"nombre": "Limitación"},
        "m": {"nombre": "Magnitud medida"},
        "c": {"nombre": "Causa"},
        "s": {"nombre": "Corrección posible"},
    })

    E.seccion("03", "Errores detectados durante el desarrollo",
              "Se documentan porque condicionaron conclusiones que ya se habían escrito.")
    E.tabla(pd.DataFrame([
        {"e": "Ventanas no comparables en la validación",
         "d": "Medían periodos distintos; la de 2,000 sesiones omitía marzo de 2020.",
         "i": "La conclusión cambió de 2,000 a 250 sesiones"},
        {"e": "El GARCH producía persistencia sobre ruido blanco",
         "d": "Límite inferior de β mal fijado, y β no identificado cuando α ≈ 0.",
         "i": "Se añadió el diagnóstico de agrupamiento"},
        {"e": "Conteo de ausencias por año con countDistinct",
         "d": "Omitía los huecos previos a la fecha de listado. Detectado por Miguel Herrera.",
         "i": "5,156 contra 54,479 ausencias reales"},
        {"e": "Detección de datos corruptos por frecuencia",
         "d": "Excluía eventos reales; GameStop subió 1,000% efectivamente.",
         "i": "Sustituida por detección de reversión: de 16 a 2 casos"},
        {"e": "stack() en pandas 3.0 conserva los nulos",
         "d": "Cambio de comportamiento respecto a versiones anteriores.",
         "i": "412,335 falsos eventos extremos en lugar de 20"},
        {"e": "γ confundida con el error del modelo de beta constante",
         "d": "γ es el salto de calma a tensión; la beta única ya absorbe parte.",
         "i": "Habría exagerado el error: el real es el 38% de γ"},
        {"e": "La asignación equiponderada caía en mínima varianza",
         "d": "La interfaz enviaba el nombre con mayúscula y el cálculo comparaba en minúscula.",
         "i": "Una opción del terminal no hacía nada, sin aviso"},
    ]), {
        "e": {"nombre": "Error"},
        "d": {"nombre": "Descripción"},
        "i": {"nombre": "Impacto"},
    })

    E.seccion("04", "Arquitectura",
              "Todo el análisis reside en src/, cubierto por 307 pruebas. El terminal "
              "únicamente presenta: no contiene lógica propia.")
    E.tabla(pd.DataFrame([
        {"m": "src/data.py", "f": "Extracción: Yahoo Finance, Wikipedia, FRED"},
        {"m": "src/quality.py", "f": "Diagnóstico de calidad y detección de datos corruptos"},
        {"m": "src/returns.py", "f": "Panel balanceado y rendimientos logarítmicos"},
        {"m": "src/analysis.py", "f": "Correlación y agregación sectorial"},
        {"m": "src/risk.py", "f": "Covarianza, encogimiento y descomposición del riesgo"},
        {"m": "src/validation.py", "f": "Validación hacia adelante"},
        {"m": "src/volatility.py", "f": "GARCH(1,1) y pronóstico de volatilidad"},
        {"m": "src/nonlinear.py", "f": "Información mutua y dependencia de cola"},
        {"m": "src/cofactors.py", "f": "Beta condicional al régimen"},
        {"m": "src/stress.py", "f": "VaR, Expected Shortfall y escenarios de crisis"},
        {"m": "src/network.py", "f": "Árbol de expansión mínima y concentradores"},
        {"m": "databricks/01–06", "f": "Equivalente a escala de 1,500 activos (PySpark)"},
        {"m": "dashboard/", "f": "Este terminal: estilo, datos y vistas"},
    ]), {"m": {"nombre": "Módulo"}, "f": {"nombre": "Función"}})

    E.seccion("05", "Glosario", "Todos los términos empleados en el terminal.")
    gl = pd.DataFrame(
        [{"t": k, "d": v} for k, v in sorted(E.GLOSARIO.items())]
    )
    E.tabla(gl, {
        "t": {"nombre": "Término"},
        "d": {"nombre": "Definición"},
    })


# ==========================================================================
# Navegación
# ==========================================================================
# Cada vista declara que insumo necesita, y la barra lateral muestra solo los
# controles de ese insumo. "cartera" es compartida: el motor de riesgo y las
# pruebas de estres hablan siempre de la misma, que es lo que convierte al
# terminal en una herramienta en lugar de siete informes sueltos.
VISTAS = {
    "Resumen ejecutivo":         (v_resumen, None),
    "Estructura de correlación": (v_correlacion, "muestra"),
    "Red del mercado":           (v_red, "muestra"),
    "Motor de riesgo":           (v_riesgo, "cartera"),
    "Volatilidad y pronóstico":  (v_volatilidad, None),
    "Comportamiento en crisis":  (v_crisis, "muestra"),
    "Pruebas de estrés":         (v_estres, "cartera"),
    "Validación del modelo":     (v_validacion, "muestra"),
    "Metodología":               (v_metodo, None),
}

CARTERA_INICIAL = ["AAPL", "JPM", "XOM", "KO", "NVDA", "SO", "PG", "UNH"]

# Tope de la muestra por vista, con su motivo. No es uno solo porque el costo
# no es el mismo: medido sobre el panel de 2,933 sesiones, recorrer los 104,196
# pares del universo completo toma 3 segundos, mientras que la validacion hacia
# adelante sobre esos mismos activos toma mas de tres minutos.
#
# En la validacion hay ademas un limite estadistico y no solo de tiempo: con N
# activos la covarianza muestral necesita mas de N observaciones, asi que una
# ventana de 125 sesiones con 457 activos tiene menos datos que parametros. El
# encogimiento de Ledoit-Wolf la vuelve invertible, pero el resultado es ruido.
LIMITES = {
    "Estructura de correlación": (457, "El recorrido completo de pares toma unos 3 segundos."),
    "Red del mercado": (
        457,
        "El árbol es inmediato; lo que cuesta es colocar los nodos para "
        "dibujarlos, y aun así el universo completo tarda segundo y medio.",
    ),
    "Comportamiento en crisis": (457, "El recorrido completo de pares toma unos 3 segundos."),
    "Validación del modelo": (
        250,
        "Cada ventana reconstruye y evalúa la cartera decenas de veces, así que "
        "el costo crece mucho más rápido: 250 activos tardan unos 40 segundos y "
        "el universo completo pasaría de tres minutos. Por encima de ~125 "
        "activos, además, la ventana más corta tiene menos observaciones que "
        "parámetros y su estimación deja de ser confiable.",
    ),
}

with st.sidebar:
    E.html(
        '<div class="logo"><div class="logo-n">Correlación y Riesgo</div>'
        '<div class="logo-s">S&P 500 · Terminal</div></div>',
        )
    E.html('<div class="lat-tit">Vistas</div>')
    eleccion = st.radio("Vistas", list(VISTAS), label_visibility="collapsed")

    vista, necesita = VISTAS[eleccion]
    n_activos, cartera = 40, None

    if necesita == "muestra":
        universo = D.cargar()[0].shape[1]
        tope, motivo = LIMITES[eleccion]
        tope = min(tope, universo)

        E.html('<div class="lat-tit">Parámetros</div>')
        # El paso de 10 hace que Streamlit redondee el extremo hacia arriba, de
        # modo que el deslizador puede ofrecer un numero mayor al universo. El
        # valor efectivo se acota aqui para que lo que se muestra coincida con
        # lo que se calcula.
        elegido = st.slider("Activos en la muestra", 20, tope, min(60, tope), step=10)
        n_activos = min(elegido, universo)

        completo = n_activos >= universo
        E.html(
            f'<div class="lat-pie">'
            + (f'<b>Universo completo</b>: los {universo} activos con historia '
               f'continua, <b>{universo * (universo - 1) // 2:,}</b> pares.'
               if completo else
               f'Los pares crecen con el cuadrado del universo: <b>{n_activos}</b> '
               f'activos son <b>{n_activos * (n_activos - 1) // 2:,}</b> pares.')
            + f'<br><br>El máximo de esta vista es <b>{tope}</b>. {motivo}</div>')

    elif necesita == "cartera":
        universo = sorted(D.cargar()[0].columns)
        E.html('<div class="lat-tit">Cartera</div>')
        posiciones = st.multiselect(
            "Posiciones", universo,
            default=[x for x in CARTERA_INICIAL if x in universo] or universo[:5])
        metodo = st.radio("Asignación", [D.MINIMA_VARIANZA, D.EQUIPONDERADO])
        encoger = st.toggle(
            "Encogimiento Ledoit-Wolf", value=True,
            help="Mezcla la covarianza observada con una versión estructurada y "
                 "estable. Evita que la optimización persiga ruido cuando hay "
                 "pocos datos por activo.")
        cartera = {"posiciones": posiciones, "metodo": metodo, "encoger": encoger}
        E.html(
            '<div class="lat-pie">La misma cartera alimenta el <b>motor de riesgo</b> '
            'y las <b>pruebas de estrés</b>.</div>')

    E.html(
        '<div class="lat-pie">Universidad Panamericana · Datos Masivos<br>'
        'Rissotto · Herrera · de la Mora · del Valle<br><br>'
        'Análisis en <b>src/</b> · <b>307 pruebas</b><br>'
        'Pasa el cursor sobre los términos subrayados para ver su definición.</div>')

if necesita == "muestra":
    vista(n_activos)
elif necesita == "cartera":
    vista(cartera)
else:
    vista()
