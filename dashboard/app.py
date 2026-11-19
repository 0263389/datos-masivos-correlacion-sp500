"""Tablero del motor de correlacion y riesgo del S&P 500.

Responde el ultimo pedido del Entregable 3 -- "que el usuario tenga donde
verlo" -- y sirve de recorrido por todo el proyecto: cada pagina corresponde a
una pregunta que se respondio en algun entregable, con el numero a la vista y
la explicacion de como se llego a el.

Se ejecuta con::

    streamlit run dashboard/app.py

Toda la logica de analisis vive en ``src/``, que es la capa probada. Este
archivo solo dibuja.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import datos as D

st.set_page_config(page_title="Motor de riesgo S&P 500", page_icon="📊",
                   layout="wide")

VERDE, ROJO, AZUL, GRIS = "#2e7d32", "#c62828", "#1565c0", "#9e9e9e"


def figura(alto: float = 3.4):
    fig, ax = plt.subplots(figsize=(9, alto))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def pct(x: float) -> str:
    return f"{x:.2%}"


# ==========================================================================
# Portada
# ==========================================================================
def portada():
    st.title("Motor de correlacion y riesgo · S&P 500")
    st.caption(
        "Proyecto de Datos Masivos · Universidad Panamericana · "
        "Franco Rissotto, Miguel Herrera, Octavio de la Mora, Paolo del Valle"
    )

    rets, _, _ = D.cargar()
    _, params, vol, tension = D.mercado_y_garch("v1")

    c = st.columns(4)
    c[0].metric("Activos en el panel", f"{rets.shape[1]}")
    c[1].metric("Dias de historia", f"{rets.shape[0]:,}")
    c[2].metric("Pares medibles", f"{rets.shape[1] * (rets.shape[1] - 1) // 2:,}")
    c[3].metric("Vol. del mercado hoy", pct(vol.iloc[-1]),
                delta=pct(vol.iloc[-1] - params["vol_largo_plazo"]),
                delta_color="inverse",
                help="Contra su nivel de largo plazo. Negativo = mas tranquilo de lo normal.")

    st.divider()
    st.subheader("Los cuatro hallazgos del proyecto")

    a, b = st.columns(2)
    with a:
        st.markdown(
            "#### 1 · Un sector no diversifica\n"
            "Dos acciones del mismo sector se parecen **1.39 veces mas** que dos "
            "de sectores distintos (0.4404 contra 0.3164 de correlacion media). "
            "Comprar cinco bancos no es repartir el riesgo."
        )
        st.markdown(
            "#### 3 · Un anio de historia basta\n"
            "Probando ventanas de 125 a 2,000 dias con validacion hacia adelante, "
            "la mejor es **250 dias**. Mas historia no ayuda: arrastra un mercado "
            "que ya no existe."
        )
    with b:
        st.markdown(
            "#### 2 · La diversificacion falla al caer\n"
            "La probabilidad de que dos acciones esten juntas en su peor 10% es "
            "**29.9%**, contra 26.8% de estar juntas en su mejor 10%. Por azar "
            "serian 10% las dos. Y pasa en el **84.5%** de los pares."
        )
        st.markdown(
            "#### 4 · Las betas convergen en las crisis\n"
            "La mitad de las acciones cambia su sensibilidad al mercado de forma "
            "significativa, y lo hace **hacia 1**: las defensivas dejan de serlo. "
            "Utilities pasa de beta 0.479 a 0.779."
        )

    st.divider()
    st.subheader("Volatilidad del mercado estimada dia por dia")
    st.caption(
        "GARCH(1,1) ajustado sobre el promedio equiponderado del panel. El modelo "
        "nunca supo que existio el COVID: el pico de marzo de 2020 lo encontro "
        "solo, leyendo unicamente precios."
    )

    fig, ax = figura(3.2)
    ax.plot(vol.index, vol * 100, color=AZUL, linewidth=0.8)
    ax.fill_between(vol.index, 0, vol * 100, where=tension.to_numpy(),
                    color=ROJO, alpha=0.18, label=f"dias tensos (quintil alto)")
    ax.axhline(params["vol_largo_plazo"] * 100, color=GRIS, linestyle="--",
               linewidth=1, label=f"largo plazo {params['vol_largo_plazo']:.1%}")
    ax.set_ylabel("volatilidad anualizada (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ==========================================================================
# 1. Correlaciones
# ==========================================================================
def pagina_correlaciones(n: int):
    st.header("Se parecen mas las acciones del mismo sector?")
    st.caption("Entregable 1 · la pregunta que abrio el proyecto")

    pares, comparacion = D.pares_y_sectores(n)
    mismo = comparacion.loc["Mismo sector", "media"]
    distinto = comparacion.loc["Distinto sector", "media"]

    c = st.columns(4)
    c[0].metric("Pares medidos", f"{len(pares):,}")
    c[1].metric("Mismo sector", f"{mismo:.4f}")
    c[2].metric("Distinto sector", f"{distinto:.4f}")
    c[3].metric("Razon", f"{mismo / distinto:.2f}x")

    st.caption(
        f"Medido sobre una submuestra de {n} activos ({len(pares):,} pares). "
        f"Sobre el universo completo de 503 acciones -el numero del Entregable 1- "
        f"la razon es 1.39x (0.4404 contra 0.3164). Mueve el control de la "
        f"izquierda para ver que la conclusion no depende de la muestra: cambian "
        f"los decimales, no el sentido."
    )
    st.info(
        "**Por que se correlacionan rendimientos y no precios.** Dos acciones que "
        "simplemente subieron durante once anios darian correlacion alta aunque no "
        "tengan nada que ver: ambas crecen con el tiempo. El rendimiento "
        "logaritmico `ln(P_t / P_t-1)` elimina ese nivel y deja solo el movimiento "
        "del dia, que es lo comparable."
    )

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Distribucion de las correlaciones")
        fig, ax = figura()
        for etiqueta, color in [(True, VERDE), (False, GRIS)]:
            sub = pares[pares["same_sector"] == etiqueta]["corr"]
            ax.hist(sub, bins=45, alpha=0.6, color=color, density=True,
                    label=f"{'mismo' if etiqueta else 'distinto'} sector (n={len(sub):,})")
        ax.axvline(mismo, color=VERDE, linestyle="--", linewidth=1.2)
        ax.axvline(distinto, color=GRIS, linestyle="--", linewidth=1.2)
        ax.set_xlabel("correlacion de Pearson")
        ax.set_ylabel("densidad")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with der:
        st.subheader("Resumen por grupo")
        st.dataframe(comparacion.style.format("{:.4f}", subset=["media", "mediana", "desv_est"]),
                     width="stretch")
        st.subheader("Los pares mas acoplados")
        st.dataframe(
            pares.nlargest(8, "corr")[["ticker_a", "ticker_b", "corr", "sector_a"]]
            .style.format({"corr": "{:.4f}"}),
            width="stretch", hide_index=True,
        )

    st.subheader("Matriz de correlacion, ordenada por sector")
    st.caption(
        "Los tickers estan ordenados por sector, asi que cada sector es un bloque "
        "contiguo. Los cuadros rojos a lo largo de la diagonal **son** los "
        "sectores: si el sector no importara, el color seria uniforme. La "
        "diagonal misma se oculta, porque vale 1 por definicion y aplastaria la "
        "escala de color."
    )
    matriz = D.matriz_por_sector(min(n, 60))
    # La diagonal vale 1 por definicion y, al dominar la escala de color,
    # aplasta todo lo demas contra el extremo claro. Se oculta, y los limites se
    # fijan con los percentiles de los valores reales para que el contraste
    # caiga donde estan los datos.
    valores = matriz.to_numpy().astype(float)
    np.fill_diagonal(valores, np.nan)
    lo, hi = np.nanpercentile(valores, [2, 98])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(valores, cmap="RdYlGn_r", vmin=lo, vmax=hi)
    ax.set_xticks(range(len(matriz)), matriz.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(matriz)), matriz.index, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlacion")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ==========================================================================
# 2. Motor de riesgo
# ==========================================================================
def pagina_riesgo(n: int):
    st.header("Cuanto riesgo tiene mi cartera, y de donde viene?")
    st.caption("Entregable 2 · el motor")

    rets, _, sectores = D.cargar()
    universo = D._muestra(rets.columns, n)
    por_defecto = [t for t in ["AAPL", "JPM", "XOM", "KO", "NVDA"] if t in rets.columns]

    sel = st.multiselect("Activos de la cartera", sorted(rets.columns),
                         default=por_defecto or universo[:5])
    a, b = st.columns(2)
    metodo = a.radio("Pesos", ["minima varianza", "equiponderado"], horizontal=True)
    encoger = b.toggle("Encoger la covarianza (Ledoit-Wolf)", value=True,
                       help="Promedia la matriz muestral con una estructurada y "
                            "estable. Sin esto, con pocos datos por activo la "
                            "optimizacion persigue ruido.")

    if len(sel) < 2:
        st.warning("Elige al menos dos activos.")
        return

    r, delta = D.riesgo(tuple(sel), metodo, encoger)

    c = st.columns(4)
    c[0].metric("Volatilidad de la cartera", pct(r["portfolio_vol"]))
    c[1].metric("Si no se cancelara nada", pct(r["weighted_avg_vol"]))
    c[2].metric("Ratio de diversificacion", f"{r['diversification_ratio']:.3f}",
                help=f"Maximo posible con {len(sel)} activos: "
                     f"{r['max_diversification_ratio']:.3f} (raiz de n)")
    c[3].metric("Encogimiento aplicado", f"{delta:.1%}" if encoger else "—",
                help="Cuanto peso le dio Ledoit-Wolf a la matriz estructurada.")

    st.success(
        f"Juntar estos {len(sel)} activos baja la volatilidad de "
        f"**{pct(r['weighted_avg_vol'])}** a **{pct(r['portfolio_vol'])}**. "
        f"Esa diferencia **es** la diversificacion: no la da el numero de "
        f"posiciones sino que no se muevan al mismo tiempo."
    )

    tabla = pd.DataFrame({
        "activo": r["labels"],
        "sector": [sectores.get(t, "") for t in r["labels"]],
        "peso": r["weights"],
        "vol. individual": r["standalone_vol"],
        "% del riesgo": r["pct_risk"],
        "riesgo / peso": r["pct_risk"] / r["weights"],
    }).sort_values("% del riesgo", ascending=False)

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("De donde viene el riesgo")
        st.caption(
            "La contribucion al riesgo **no** es proporcional al dinero invertido. "
            "La ultima columna lo mide: por encima de 1 la posicion aporta mas "
            "riesgo que capital."
        )
        st.dataframe(
            tabla.style.format({"peso": "{:.2%}", "vol. individual": "{:.2%}",
                                "% del riesgo": "{:.2%}", "riesgo / peso": "{:.2f}x"})
            .background_gradient(subset=["riesgo / peso"], cmap="OrRd"),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Las contribuciones suman {r['pct_risk'].sum():.4%} "
            f"(error {r['check_sum']:.1e}). No es casualidad: por el teorema de "
            f"Euler el reparto es exacto, no una aproximacion."
        )
        if metodo == "minima varianza":
            st.info(
                "**Si toda la columna `riesgo / peso` da 1.00x, no es un error: es "
                "la demostracion de que la optimizacion funciono.** La condicion "
                "que define el portafolio de minima varianza es justamente que "
                "ninguna posicion aporte riesgo desproporcionado a su peso -- si "
                "alguna lo hiciera, moverle capital bajaria la volatilidad total, y "
                "entonces no seria el minimo. Cambia a **equiponderado** para ver "
                "la columna con valores distintos."
            )
    with der:
        st.subheader("Peso contra riesgo")
        fig, ax = figura(4.2)
        y = np.arange(len(tabla))
        ax.barh(y - 0.2, tabla["peso"] * 100, height=0.38, color=AZUL, label="peso")
        ax.barh(y + 0.2, tabla["% del riesgo"] * 100, height=0.38, color=ROJO,
                label="% del riesgo")
        ax.set_yticks(y, tabla["activo"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("%")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ==========================================================================
# 3. Cuanta historia
# ==========================================================================
def pagina_ventana(n: int):
    st.header("Cuantos dias de historia conviene usar?")
    st.caption("Entregable 3 · validacion hacia adelante")

    st.info(
        "**El metodo.** Se simula ser un inversionista del pasado: con solo la "
        "informacion disponible hasta ese dia se arma la cartera, se mantiene 21 "
        "dias, se mide lo que realmente paso, y se repite avanzando en el tiempo. "
        "El modelo **nunca ve el futuro**. Si lo viera, los resultados saldrian "
        "preciosos y serian falsos."
    )

    tabla = D.barrido(n)
    mejor = tabla.loc[tabla["vol_realizada"].idxmin()]

    c = st.columns(3)
    c[0].metric("Ventana optima", f"{int(mejor['ventana_dias'])} dias",
                help=f"{mejor['ventana_anios']:.1f} anios")
    c[1].metric("Volatilidad lograda", pct(mejor["vol_realizada"]))
    c[2].metric("Rotacion mensual", pct(mejor["rotacion_media"]),
                help="Cuanto hay que mover la cartera en cada rebalanceo. "
                     "Ventanas cortas reaccionan rapido pero cuestan mas operar.")

    izq, der = st.columns([3, 2])
    with izq:
        fig, ax = figura(3.6)
        ax.plot(tabla["ventana_dias"], tabla["vol_realizada"] * 100, "o-",
                color=AZUL, linewidth=1.6)
        ax.scatter([mejor["ventana_dias"]], [mejor["vol_realizada"] * 100],
                   s=150, facecolors="none", edgecolors=ROJO, linewidths=2,
                   zorder=5, label="optimo")
        ax.set_xlabel("dias de historia usados")
        ax.set_ylabel("volatilidad REAL lograda (%)")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(
            "La curva sube a la derecha: **mas historia empeora el resultado**. "
            "Los mercados cambian, y datos de hace ocho anios describen un mercado "
            "que ya no existe."
        )
    with der:
        st.dataframe(
            tabla[["ventana_dias", "ventana_anios", "vol_realizada",
                   "rotacion_media", "n_rebalanceos"]]
            .style.format({"ventana_anios": "{:.1f}", "vol_realizada": "{:.2%}",
                           "rotacion_media": "{:.1%}"}),
            width="stretch", hide_index=True,
        )
        st.warning(
            "**Un error que hubo que corregir.** La primera version comparaba "
            "ventanas que median periodos distintos: la de 250 dias alcanzaba a "
            "vivir el COVID y la de 2,000 arrancaba despues. La respuesta cambio "
            "de 2,000 a 250 dias al darle a todas el mismo punto de arranque "
            f"(dia {int(mejor['evalua_desde_dia'])})."
        )


# ==========================================================================
# 4. Volatilidad
# ==========================================================================
def pagina_volatilidad():
    st.header("Cuanto se va a mover esta accion?")
    st.caption("Entregable 3 · GARCH(1,1)")

    rets, _, _ = D.cargar()
    ticker = st.selectbox("Activo", sorted(rets.columns),
                          index=sorted(rets.columns).index("AAPL")
                          if "AAPL" in rets.columns else 0)
    params, vol, pronostico = D.garch_de(ticker)

    c = st.columns(5)
    c[0].metric("alpha · reaccion", f"{params['alpha']:.3f}",
                help="Cuanto mueve la prediccion el golpe de hoy.")
    c[1].metric("beta · memoria", f"{params['beta']:.3f}",
                help="Cuanto arrastra de la prediccion de ayer.")
    c[2].metric("persistencia", f"{params['persistencia']:.3f}",
                help="alpha + beta. Cuanto duran las rachas.")
    c[3].metric("vol. de largo plazo", pct(params["vol_largo_plazo"]))
    c[4].metric("hay agrupamiento?", "SI" if params["hay_agrupamiento"] else "NO",
                help=f"Mejora sobre volatilidad constante: "
                     f"{params['mejora_vs_constante']:.0f}. Umbral: 3.")

    if not params["hay_agrupamiento"]:
        st.error(
            "El modelo **no** le gana a suponer volatilidad constante en esta "
            "serie, asi que su persistencia no significa nada. El diagnostico "
            "existe justamente para no reportar un numero vacio."
        )

    st.info(
        "**La idea.** La volatilidad viene en rachas: despues de un dia movido es "
        "mas probable que venga otro, como el clima. GARCH lo escribe como "
        "`varianza de maniana = omega + alpha x (golpe de hoy)^2 + "
        "beta x (varianza de hoy)`."
    )

    st.subheader("Lo que el modelo estimaba cada dia")
    realizada = rets[ticker].rolling(21).std() * np.sqrt(252)
    fig, ax = figura(3.4)
    ax.plot(realizada.index, realizada * 100, color=GRIS, linewidth=0.7,
            label="volatilidad realizada (21 dias)")
    ax.plot(vol.index, vol * 100, color=ROJO, linewidth=1.0,
            label="estimada por GARCH")
    ax.axhline(params["vol_largo_plazo"] * 100, color=AZUL, linestyle="--",
               linewidth=1, label="largo plazo")
    ax.set_ylabel("volatilidad anualizada (%)")
    ax.legend(frameon=False, fontsize=8)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    maximo = vol.idxmax()
    st.success(
        f"El maximo historico de volatilidad estimada para **{ticker}** cae el "
        f"**{maximo.date()}** ({vol.max():.1%}). Nadie le dijo al modelo que "
        f"existia el COVID."
    )

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Prediccion para el proximo anio")
        st.caption(
            "Es lo que un promedio movil no puede dar: la prediccion **regresa "
            "sola** hacia el nivel normal del activo, y la persistencia decide "
            "que tan rapido."
        )
        fig, ax = figura(3.0)
        ax.plot(range(1, len(pronostico) + 1), pronostico * 100, color=AZUL,
                linewidth=1.8)
        ax.axhline(params["vol_largo_plazo"] * 100, color=GRIS, linestyle="--",
                   linewidth=1, label="largo plazo")
        ax.set_xlabel("dias hacia adelante")
        ax.set_ylabel("volatilidad anualizada (%)")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with der:
        st.subheader("Horizontes")
        st.dataframe(pd.DataFrame({
            "horizonte": ["manana", "1 semana", "1 mes", "6 meses", "1 anio"],
            "volatilidad": [pronostico[i] for i in (0, 4, 20, 125, 251)],
        }).style.format({"volatilidad": "{:.2%}"}), width="stretch",
            hide_index=True)
        st.caption(
            "**Limitacion medida.** El modelo subestima los picos de crisis "
            "porque supone errores normales y los mercados tienen colas mas "
            "gordas. Se corregiria con una t de Student."
        )


# ==========================================================================
# 5. Relaciones no lineales
# ==========================================================================
def pagina_no_lineal(n: int):
    st.header("Hay relaciones que la correlacion no ve?")
    st.caption("Entregable 3 · informacion mutua y dependencia de cola")

    tabla, piso = D.no_lineal(n)
    supera = (tabla["brecha"] > piso).mean()

    st.info(
        "**El problema.** La correlacion de Pearson es la pendiente de la mejor "
        "**recta**. Si la relacion no es una recta, no la ve. La informacion mutua "
        "si: mide cuanta incertidumbre sobre un activo desaparece al conocer el "
        "otro, sin suponer ninguna forma."
    )

    c = st.columns(4)
    c[0].metric("Pares medidos", f"{len(tabla):,}")
    c[1].metric("Brecha promedio", f"{tabla['brecha'].mean():.4f}")
    c[2].metric("Piso del metodo", f"{piso:.4f}",
                help="Brecha que aparece entre series SIN relacion. Por debajo "
                     "de esto, una brecha no es evidencia de nada.")
    c[3].metric("Pares que lo superan", f"{supera:.1%}",
                help="Por construccion el 5% lo superaria por azar.")

    # El piso es el percentil 95 bajo independencia, asi que el 5% lo supera por
    # construccion. Lo informativo es la razon contra ese 5%, no el porcentaje.
    razon = supera / 0.05
    if razon < 3:
        st.warning(
            f"**Resultado negativo, y hay que reportarlo asi.** El {supera:.1%} de "
            f"los pares supera el piso del metodo. Como el piso es el percentil 95 "
            f"bajo independencia, el 5% lo superaria **por construccion**, asi que "
            f"lo observado es apenas {razon:.1f} veces el azar. La informacion "
            f"mutua no encuentra evidencia solida de relaciones curvas en "
            f"rendimientos diarios: la no linealidad de este mercado no esta en la "
            f"forma de la relacion entre dos acciones. Lo que sigue muestra donde "
            f"si esta."
        )
    else:
        st.success(
            f"El {supera:.1%} de los pares supera el piso, {razon:.1f} veces el 5% "
            f"que lo superaria por construccion. Hay dependencia curva medible en "
            f"una minoria de pares."
        )

    st.divider()
    st.subheader("Donde si esta la no linealidad: en las colas")

    caidas, subidas = tabla["prob_caidas"].mean(), tabla["prob_subidas"].mean()
    c = st.columns(4)
    c[0].metric("Caer juntos", f"{caidas:.1%}", delta=f"{caidas - 0.10:+.1%} vs azar")
    c[1].metric("Subir juntos", f"{subidas:.1%}", delta=f"{subidas - 0.10:+.1%} vs azar")
    c[2].metric("Asimetria", f"{caidas - subidas:+.1%}")
    c[3].metric("Pares que caen mas juntos", f"{(tabla['asimetria'] > 0).mean():.1%}")

    izq, der = st.columns([3, 2])
    with izq:
        fig, ax = figura(3.4)
        ax.hist(tabla["prob_caidas"], bins=40, alpha=0.62, color=ROJO,
                label=f"caer juntos (media {caidas:.1%})")
        ax.hist(tabla["prob_subidas"], bins=40, alpha=0.62, color=VERDE,
                label=f"subir juntos (media {subidas:.1%})")
        ax.axvline(0.10, color="black", linestyle="--", linewidth=1.2,
                   label="si fueran independientes (10%)")
        ax.set_xlabel("P(el otro tambien esta en su 10% extremo)")
        ax.set_ylabel("pares")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.error(
            f"**El hallazgo del modulo.** La diversificacion se debilita en las "
            f"caidas de forma sistematica: pasa en el "
            f"{(tabla['asimetria'] > 0).mean():.1%} de los pares. Y es algo que un "
            f"solo coeficiente de correlacion **no puede expresar**, porque es el "
            f"mismo numero para toda la nube de puntos."
        )
    with der:
        st.subheader("Mayor brecha no lineal")
        st.dataframe(
            tabla.head(10)[["activo_a", "activo_b", "pearson",
                            "corr_equivalente", "brecha"]]
            .style.format({"pearson": "{:.3f}", "corr_equivalente": "{:.3f}",
                           "brecha": "{:.3f}"}),
            width="stretch", hide_index=True,
        )
        st.subheader("Mas asimetricos")
        st.dataframe(
            tabla.nlargest(10, "asimetria")[["activo_a", "activo_b",
                                             "prob_caidas", "prob_subidas"]]
            .style.format({"prob_caidas": "{:.1%}", "prob_subidas": "{:.1%}"}),
            width="stretch", hide_index=True,
        )


# ==========================================================================
# 6. Cofactores
# ==========================================================================
def pagina_cofactores():
    st.header("La beta cambia cuando el mercado se tensa?")
    st.caption("Entregable 3 · cofactores")

    st.info(
        "**Que es un cofactor.** Una variable que **modifica el efecto de otra** "
        "en lugar de sumar el suyo. Se mete como producto:\n\n"
        "`r_activo = alfa + beta x r_mercado + gamma x (r_mercado x tension)`\n\n"
        "En calma la sensibilidad es `beta`; en tension es `beta + gamma`. "
        "Entonces **gamma es el escalon**. El regimen no se elige a mano: lo "
        "define el GARCH de la pagina anterior."
    )

    tabla = D.cofactores()
    _, _, _, tension = D.mercado_y_garch("v1")
    sig = tabla[tabla["significativo"]]
    convergencia = np.corrcoef(tabla["beta_calma"], tabla["gamma"])[0, 1]

    c = st.columns(4)
    c[0].metric("Activos analizados", f"{len(tabla)}")
    c[1].metric("Con gamma significativa", f"{tabla['significativo'].mean():.1%}",
                help="Prueba al 5% con errores estandar robustos de White.")
    c[2].metric("Dias tensos", f"{tension.sum():,}",
                help=f"{tension.mean():.0%} del panel, el quintil alto de "
                     f"volatilidad condicional del mercado.")
    c[3].metric("corr(beta_calma, gamma)", f"{convergencia:+.3f}",
                help="Muy negativa = las betas bajas suben y las altas bajan.")

    st.success(
        f"**El hallazgo principal del entregable.** Las betas **convergen hacia 1** "
        f"en las crisis. La dispersion de betas cae de "
        f"{tabla['beta_calma'].std():.3f} a {tabla['beta_tension'].std():.3f} "
        f"({1 - tabla['beta_tension'].std() / tabla['beta_calma'].std():.1%} menos), "
        f"y la distancia promedio a 1 baja de "
        f"{(tabla['beta_calma'] - 1).abs().mean():.3f} a "
        f"{(tabla['beta_tension'] - 1).abs().mean():.3f}. "
        f"Las acciones defensivas **dejan de ser defensivas** justo cuando se "
        f"necesita que lo sean."
    )

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Beta en calma contra beta en tension")
        st.caption(
            "Si la beta no cambiara, todos los puntos caerian sobre la diagonal. "
            "Los puntos giran alrededor de 1: por encima de la diagonal a la "
            "izquierda, por debajo a la derecha."
        )
        fig, ax = figura(4.6)
        colores = np.where(tabla["significativo"], ROJO, GRIS)
        ax.scatter(tabla["beta_calma"], tabla["beta_tension"], s=14,
                   c=colores, alpha=0.65, edgecolors="none")
        lim = [tabla[["beta_calma", "beta_tension"]].min().min() - 0.1,
               tabla[["beta_calma", "beta_tension"]].max().max() + 0.1]
        ax.plot(lim, lim, color="black", linewidth=1, linestyle="--",
                label="sin cambio")
        ax.axhline(1, color=AZUL, linewidth=0.8, alpha=0.6)
        ax.axvline(1, color=AZUL, linewidth=0.8, alpha=0.6)
        ax.set_xlabel("beta en calma")
        ax.set_ylabel("beta en tension")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with der:
        st.subheader("Por quintil de beta en calma")
        tabla = tabla.copy()
        tabla["quintil"] = pd.qcut(tabla["beta_calma"], 5, labels=[1, 2, 3, 4, 5])
        q = tabla.groupby("quintil", observed=True).agg(
            n=("ticker", "size"), calma=("beta_calma", "mean"),
            tension=("beta_tension", "mean"), gamma=("gamma", "mean"),
        )
        st.dataframe(q.style.format({"calma": "{:.3f}", "tension": "{:.3f}",
                                     "gamma": "{:+.3f}"}),
                     width="stretch")
        st.caption(
            "La columna gamma es monotona: **+** en las defensivas, **−** en las "
            "agresivas. Eso es convergencia, no ruido."
        )
        st.warning(
            "**La gamma promedio es exactamente cero, y es una identidad.** El "
            "mercado se define como el promedio de estas mismas acciones, asi que "
            "la beta promedio contra el vale 1 en cualquier submuestra. El "
            "agregado **no puede** revelar nada: toda la informacion esta en la "
            "dispersion."
        )

    st.divider()
    st.subheader("Por sector: quien pierde su defensividad")
    s = tabla.groupby("sector").agg(
        n=("ticker", "size"), beta_calma=("beta_calma", "mean"),
        beta_tension=("beta_tension", "mean"), gamma=("gamma", "mean"),
        significativas=("significativo", "mean"),
    ).sort_values("gamma", ascending=False)

    fig, ax = figura(4.0)
    y = np.arange(len(s))
    ax.barh(y, s["gamma"], color=np.where(s["gamma"] > 0, ROJO, VERDE), alpha=0.8)
    ax.set_yticks(y, s.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("gamma promedio (cambio de beta al tensarse el mercado)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(
        s.style.format({"beta_calma": "{:.3f}", "beta_tension": "{:.3f}",
                        "gamma": "{:+.3f}", "significativas": "{:.0%}"})
        .background_gradient(subset=["gamma"], cmap="RdYlGn_r"),
        width="stretch",
    )

    st.divider()
    st.subheader("La forma del escalon")
    a, b = st.columns([1, 3])
    elegido = a.selectbox("Activo", tabla.sort_values("gamma", ascending=False)["ticker"])
    grupos = a.slider("Niveles de tension", 3, 8, 5)
    esc = D.escalones(elegido, grupos)

    with b:
        st.caption(
            "Version libre del modelo: se estima la beta por separado dentro de "
            "cada nivel de volatilidad, sin imponer que el cambio sea un salto "
            "unico. Si el ultimo nivel se despega, el escalon es real."
        )
        fig, ax = figura(3.2)
        ax.errorbar(esc["nivel"], esc["beta"], yerr=1.96 * esc["ee_robusto"],
                    fmt="o-", color=AZUL, capsize=4, linewidth=1.6)
        ax.set_xlabel("nivel de tension del mercado (1 = mas tranquilo)")
        ax.set_ylabel(f"beta de {elegido}")
        ax.set_xticks(esc["nivel"])
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    st.dataframe(
        esc.style.format({"beta": "{:.3f}", "ee_robusto": "{:.3f}",
                          "r2": "{:.3f}", "vol_mercado": "{:.1%}"}),
        width="stretch", hide_index=True,
    )


# ==========================================================================
# 7. Como se hizo
# ==========================================================================
def pagina_metodo():
    st.header("Como se hizo, y que no se escondio")
    st.caption("decisiones de metodo y limitaciones medidas")

    st.subheader("Decisiones que cambian el resultado")
    st.dataframe(pd.DataFrame([
        {"decision": "Rendimientos logaritmicos, no precios",
         "por que": "Dos acciones que solo subieron con los anios darian correlacion alta sin tener relacion. El rendimiento elimina el nivel.",
         "efecto": "evita correlacion espuria"},
        {"decision": "Precios ajustados por splits y dividendos",
         "por que": "Un split 4:1 se veria como una caida de -75% en un dia.",
         "efecto": "evita falsos eventos extremos"},
        {"decision": "Panel balanceado, sin rellenar huecos",
         "por que": "Rellenar un precio faltante inventa un dia de rendimiento cero, que es informacion falsa.",
         "efecto": "menos activos, datos reales"},
        {"decision": "Encogimiento de Ledoit-Wolf",
         "por que": "Con N activos hay N(N+1)/2 parametros y solo T dias. La matriz muestral queda mal condicionada e invertirla amplifica el ruido.",
         "efecto": "carteras que sobreviven fuera de muestra"},
        {"decision": "Minima varianza sin ventas en corto",
         "por que": "La version libre llego a posiciones de -155%: explota errores de estimacion. Es optima e imposible de implementar.",
         "efecto": "regularizacion implicita"},
        {"decision": "Errores estandar robustos de White",
         "por que": "El propio GARCH demostro que la varianza no es constante. Los clasicos la subestiman 1.54 veces e inflarian la significancia.",
         "efecto": "significancia honesta"},
        {"decision": "Regimen definido por el GARCH, no a mano",
         "por que": "Elegir el corte a ojo invita a escoger el que da el resultado deseado.",
         "efecto": "no hay grado de libertad oculto"},
    ]), width="stretch", hide_index=True)

    st.subheader("Limitaciones, medidas y no estimadas a ojo")
    st.dataframe(pd.DataFrame([
        {"limitacion": "GARCH subestima los picos de crisis",
         "magnitud": "91% estimado contra 114% realizado en marzo 2020",
         "causa": "supone errores normales; los mercados tienen colas mas gordas",
         "arreglo": "distribucion t de Student (agrega un parametro)"},
        {"limitacion": "GARCH apenas le gana al metodo simple a un dia",
         "magnitud": "gana en 3 de 5 activos; 0.678 contra 0.625 en XOM",
         "causa": "resultado conocido de la literatura a horizonte corto",
         "arreglo": "ninguno: su ventaja es el pronostico a varios dias"},
        {"limitacion": "La informacion mutua no detecta no linealidad",
         "magnitud": "6.0% de los pares supera el piso, contra 5% por azar",
         "causa": "discretizar pierde informacion y el sesgo residual la agrega",
         "arreglo": "ninguno: el resultado negativo es el hallazgo"},
        {"limitacion": "El umbral de tension usa la muestra completa",
         "magnitud": "0.25% de los dias cambia de lado al recalcular",
         "causa": "el cuantil conoce toda la historia; la volatilidad no",
         "arreglo": "expandiendo=True, estrictamente causal"},
        {"limitacion": "El mercado es un promedio equiponderado",
         "magnitud": "la beta promedio es 1 por construccion",
         "causa": "sin capitalizaciones dia por dia es la aproximacion honesta al primer componente",
         "arreglo": "ponderar por capitalizacion historica"},
    ]), width="stretch", hide_index=True)

    st.subheader("Errores que se encontraron y se corrigieron")
    st.markdown(
        "- **Ventanas no comparables** en la validacion hacia adelante: median "
        "periodos distintos, y la de 2,000 dias se saltaba el COVID. La respuesta "
        "cambio de 2,000 a 250 dias al corregirlo.\n"
        "- **GARCH inventaba rachas** sobre ruido puro (persistencia 0.99). Dos "
        "causas: un limite mal puesto, y que beta no esta identificado cuando "
        "alpha es cero. Se agrego un diagnostico que lo detecta.\n"
        "- **Conteo de faltantes por anio** con `countDistinct` perdia los huecos "
        "previos al listado: 5,156 contra 54,479 reales. Lo detecto Miguel y se "
        "verifico empiricamente.\n"
        "- **Deteccion de datos corruptos** por frecuencia excluia eventos reales "
        "(GameStop subio 1,000% de verdad). Se reemplazo por deteccion de "
        "reversion: las corrupciones bajaron de 16 a 2.\n"
        "- **`.stack()` en pandas 3.0** conserva los nulos: 412,335 falsos eventos "
        "extremos en lugar de 20."
    )

    st.subheader("El codigo")
    st.dataframe(pd.DataFrame([
        {"archivo": "src/data.py", "responde": "de donde salen los datos"},
        {"archivo": "src/quality.py", "responde": "son confiables?"},
        {"archivo": "src/returns.py", "responde": "precios a rendimientos"},
        {"archivo": "src/analysis.py", "responde": "se mueven juntas?"},
        {"archivo": "src/risk.py", "responde": "cuanto riesgo tiene la cartera?"},
        {"archivo": "src/validation.py", "responde": "cuanta historia usar?"},
        {"archivo": "src/volatility.py", "responde": "cuanto se va a mover?"},
        {"archivo": "src/nonlinear.py", "responde": "hay relaciones que la correlacion no ve?"},
        {"archivo": "src/cofactors.py", "responde": "la beta cambia con el regimen?"},
        {"archivo": "databricks/01-06", "responde": "lo mismo a escala de 1,500 activos"},
        {"archivo": "tests/ (111 pruebas)", "responde": "todo lo anterior es cierto?"},
    ]), width="stretch", hide_index=True)


# ==========================================================================
# Navegacion
# ==========================================================================
PAGINAS = {
    "Portada": None,
    "1 · Correlaciones": "n",
    "2 · Motor de riesgo": "n",
    "3 · Cuanta historia": "n",
    "4 · Volatilidad (GARCH)": None,
    "5 · Relaciones no lineales": "n",
    "6 · Cofactores": None,
    "7 · Como se hizo": None,
}

with st.sidebar:
    st.title("📊 Motor S&P 500")
    eleccion = st.radio("Seccion", list(PAGINAS), label_visibility="collapsed")
    st.divider()
    if PAGINAS[eleccion] == "n":
        n_activos = st.slider(
            "Activos en el universo", 20, 120, 40, step=10,
            help="Los pares crecen con el cuadrado: 40 activos son 780 pares, "
                 "120 son 7,140. Subelo para mas evidencia, bajalo para mas "
                 "velocidad.",
        )
    else:
        n_activos = 40
    st.divider()
    st.caption(
        "Todo el analisis vive en `src/`, con 111 pruebas. Este tablero solo "
        "dibuja: no contiene logica propia."
    )

if eleccion == "Portada":
    portada()
elif eleccion.startswith("1"):
    pagina_correlaciones(n_activos)
elif eleccion.startswith("2"):
    pagina_riesgo(n_activos)
elif eleccion.startswith("3"):
    pagina_ventana(n_activos)
elif eleccion.startswith("4"):
    pagina_volatilidad()
elif eleccion.startswith("5"):
    pagina_no_lineal(n_activos)
elif eleccion.startswith("6"):
    pagina_cofactores()
else:
    pagina_metodo()
