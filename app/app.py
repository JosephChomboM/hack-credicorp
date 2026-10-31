"""Dashboard de decisión · Value Finders.

Responde la pregunta del reto: ¿qué decisión de inversión recomendamos a este cliente hoy,
a partir de la información disponible, y qué evidencia permite defenderla?

Ejecutar desde esta carpeta:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Decisión de inversión", page_icon="📈", layout="wide")

# Paleta categórica fija: cada activo conserva su color en todos los gráficos
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = dict(zip(E.ASSETS, PALETTE))
CASH_COLOR, NEUTRAL, INK_2, GRID = "#8a8984", "#c3c2b7", "#5f5e5a", "#e1e0d9"
CHOSEN = "CL_02"


# ---------- utilidades ----------
def pct(x, d=1, signed=False):
    s = f"{x * 100:+.{d}f}" if signed else f"{x * 100:.{d}f}"
    return s.replace(".", ",") + "%"


def pts(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", ",") + " pts"


def style(fig, height=340, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, separators=",.", hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10), font=dict(size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0) if legend else None, showlegend=legend,
        title=dict(font=dict(size=14)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def label(a, ref):
    return f"{a[3:]} {ref.at[a, 'name']}"


# ---------- datos (cacheados) ----------
@st.cache_data(show_spinner="Cargando data_clean…")
def get_data():
    return E.load_data()


@st.cache_data(show_spinner="Simulando 2.000 años posibles…")
def get_bootstrap(client_id, cash):
    d = get_data()
    R = E.local_returns(d["r_usd"], d["macro"], d["rules"][client_id]["currency"])
    w = E.portfolio_weights(d["weights"], client_id, cash)
    ew = pd.Series((1 - cash) / 8, E.ASSETS)
    return E.block_bootstrap(R, w), E.block_bootstrap(R, ew)


@st.cache_data(show_spinner="Evaluando carteras alternativas…")
def get_feasibility(client_id, cash, pct_per_point):
    d = get_data()
    rules = d["rules"][client_id]
    R = E.local_returns(d["r_usd"], d["macro"], rules["currency"])
    return E.feasibility(R, E.expected_returns(d["close"], pct_per_point), rules, cash, d["ref"])


@st.cache_data(show_spinner="Leyendo prices_expanded_daily.csv…")
def get_expanded():
    return E.load_expanded_daily()


d = get_data()
close, r_usd, macro, ref, clients = d["close"], d["r_usd"], d["macro"], d["ref"], d["clients"]
RULES = d["rules"]

# ---------- barra lateral ----------
with st.sidebar:
    st.header("Configuración")
    cid = st.selectbox(
        "Cliente", list(RULES), index=list(RULES).index(CHOSEN),
        format_func=lambda c: f"{c} · {clients.at[c, 'profile_name']}" + (" ★" if c == CHOSEN else ""),
        help="★ = cliente elegido para la decisión. El motor también calcula los otros perfiles.")
    rules = RULES[cid]
    cur, lim = rules["currency"], rules["dd_limit"]

    st.subheader("Escenario del twist")
    st.caption("Efectivo desde el 17-ago-2026, el primer día hábil sin precios.")
    rate_bp = st.slider("TW_01 · Alza de la tasa a 10 años (pb)", 0, 130, E.BASE_RATE_BP, 5)
    fx_pct = st.slider("TW_03 · Depreciación del COP (%)", 0, 15, 8,
                       disabled=not rules["fx_shock_applies"],
                       help="Solo afecta a clientes que miden en COP.")
    rec_pct = st.slider("Si el COP se recupera antes del retiro (%)", 0, 15, 8,
                        disabled=not rules["fx_shock_applies"],
                        help="Prueba qué se pierde si la liquidez no se convierte hoy a COP.")
    cash = st.slider("Liquidez en la moneda del cliente (%)", 0, 60, int(rules["cash"] * 100), 5,
                     help="TW_02: CL_02 necesitará el 30% en 6 meses. La liquidez rinde 0% (supuesto conservador).") / 100
    st.divider()
    st.caption("Fuente: data_clean/ (tablas del notebook 01; carteras y robustez del notebook 04) y dataset_twist.csv.")

ppp = E.RATE_PCT_PER_POINT * rate_bp / E.BASE_RATE_BP
fx_shock = fx_pct / 100 if rules["fx_shock_applies"] else 0.0
rec = rec_pct / 100 if rules["fx_shock_applies"] else 0.0

R = E.local_returns(r_usd, macro, cur)
mu = E.expected_returns(close, ppp)
w = E.portfolio_weights(d["weights"], cid, cash)
ew = pd.Series((1 - cash) / 8, E.ASSETS)
m, m_ew = E.portfolio_metrics(R, w, mu), E.portfolio_metrics(R, ew, mu)
(dd_b, ret_b), (dd_ew_b, _) = get_bootstrap(cid, cash)
# Con la liquidez del 04 se muestran sus cifras exportadas; si se mueve la liquidez, se recalculan en vivo
nb_case = abs(cash - rules["cash"]) < 1e-9 and cid in d["robustness"].index
if nb_case:
    rob = d["robustness"].loc[cid]
    p_breach, p_breach_ew = rob["P(caída > límite) recomendado %"] / 100, rob["P(caída > límite) igual peso %"] / 100
else:
    p_breach, p_breach_ew = (np.abs(dd_b) > lim).mean(), (np.abs(dd_ew_b) > lim).mean()
twist = E.twist_impact(w, cash, ppp, fx_shock)
rate_hit = float((w * E.RATE_SCALE * ppp).sum())
feas_ok, feas_all = get_feasibility(cid, cash, ppp)
shock = E.shock_size(macro, fx_shock or 0.08)
core = [a for a in w.sort_values(ascending=False).index if w[a] > 0][:3]
core_txt = " + ".join(f"{a[3:]} {ref.at[a, 'name']}" for a in core)
protected = cash * (1 + fx_shock) * rec

# ---------- encabezado y decisión ----------
tab_main, tab_quality, tab_method = st.tabs(["Decisión", "Calidad de datos", "Metodología"])

with tab_main:
    st.title("¿Qué decisión de inversión recomendamos hoy?")
    st.caption(f"{cid} · {clients.at[cid, 'profile_name']} · mide en {cur} · caída máxima tolerada {pct(lim, 0)} · "
               f"horizonte {clients.at[cid, 'horizon_months']} meses · datos al {close.index[-1]:%d-%m-%Y}")

    within = abs(m["caida_max"]) <= lim
    if cid == "CL_02":
        decision = (f"**Separar el portafolio:** {pct(cash, 0)} en **liquidez en COP**, convertida hoy, para el retiro a 6 meses, "
                    f"y {pct(1 - cash, 0)} en un **núcleo defensivo** ({core_txt}). "
                    f"{'Cumple' if within else 'No cumple'} su límite de caída de {pct(lim, 0)} y el twist le resulta "
                    f"{'favorable' if twist >= 0 else 'desfavorable'} ({pct(twist, 1, True)}).")
    else:
        decision = (f"**Invertir en el núcleo {core_txt}**" + (f" con {pct(cash, 0)} de liquidez" if cash else "") +
                    f". {'Cumple' if within else 'No cumple'} su límite de caída de {pct(lim, 0)}; "
                    f"el twist le cuesta {pct(twist, 1, True)}.")
    (st.success if within else st.error)(decision, icon="✅" if within else "⚠️")

    k = st.columns(5)
    k[0].metric(f"Caída máxima 12m ({cur})", pct(m["caida_max"]),
                f"{pts(lim + m['caida_max'])} de margen frente al límite", delta_color="off", border=True,
                help="Peor caída desde un máximo, en la moneda del cliente, con la cartera recomendada.")
    k[1].metric("Probabilidad de superar el límite", pct(p_breach),
                f"{pts(p_breach - p_breach_ew)} frente a la cartera a partes iguales", delta_color="inverse",
                border=True, help="Bootstrap por bloques de 20 días: 2.000 años simulados con la historia. "
                                  + ("Cifra de recommendation_robustness.csv (notebook 04)." if nb_case
                                     else "Recalculada en vivo con la liquidez elegida."))
    k[2].metric("Impacto del twist", pct(twist, 1, True),
                "tasa + moneda" if rules["fx_shock_applies"] else "solo tasa", delta_color="off", border=True,
                help="Shock de tasas por sector (supuesto) + depreciación del COP sobre la parte invertida en USD.")
    k[3].metric("Peor resultado a 6 meses", pct(m["peor_6m"]), f"partes iguales: {pct(m_ew['peor_6m'])}",
                delta_color="off", border=True, help="Peor ventana móvil de 126 días hábiles.")
    k[4].metric(f"Retorno 12m ({cur})", pct(m["retorno_12m"], 1, True),
                f"{pts(m['retorno_12m'] - m_ew['retorno_12m'])} frente a partes iguales", border=True,
                help="Retorno histórico de la cartera recomendada en el último año, en la moneda del cliente.")

    # ---------- 1. ¿Qué está ocurriendo? ----------
    st.header("1 · ¿Qué está ocurriendo?")
    c1, c2 = st.columns([1.8, 1])
    with c1:
        base = close / close.iloc[0] * 100
        fig = go.Figure()
        for regime, color in [("Stress", "rgba(227,73,72,.12)"), ("Recovery", "rgba(27,175,122,.12)")]:
            days = macro.index[macro.risk_regime == regime]
            if len(days):
                fig.add_vrect(x0=days.min(), x1=days.max(), fillcolor=color, line_width=0,
                              annotation_text=regime, annotation_position="top left",
                              annotation_font=dict(size=11, color=INK_2))
        for a in E.ASSETS:
            fig.add_trace(go.Scatter(x=base.index, y=base[a], name=label(a, ref), line=dict(color=COLOR[a], width=2),
                                     hovertemplate="%{y:.1f}<extra>" + label(a, ref) + "</extra>"))
        fig.add_hline(y=100, line=dict(color=NEUTRAL, width=1))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig, 380).update_layout(title="Precio de cada activo (base 100 = 1-ago-2025, USD)"),
                        width="stretch")
    with c2:
        r12 = close.iloc[-1] / close.iloc[0] - 1
        up = r12[r12 > 0].sort_values(ascending=False)
        ew_full = (1 + R[E.ASSETS] @ np.full(8, 1 / 8)).cumprod()
        fx_txt = ""
        if cur != "USD":
            chg = macro[f"usd_{cur.lower()}"].iloc[-1] / macro[f"usd_{cur.lower()}"].iloc[0] - 1
            fx_txt = (f"- El dólar {'se fortaleció' if chg > 0 else 'se debilitó'} **{pct(chg, 1, True)}** frente al {cur}: "
                      f"{'suma' if chg > 0 else 'resta'} retorno a este cliente.\n")
        st.markdown(
            f"- Solo **{len(up)} de 8** activos suben en 12 meses: "
            + ", ".join(f"**{label(a, ref)} ({pct(v, 1, True)})**" for a, v in up.items()) + ".\n"
            f"- Una cartera a partes iguales rinde **{pct(ew_full.iloc[-1] - 1, 1, True)}** "
            f"en {cur} y llega a caer **{pct(E.max_drawdown(ew_full))}**.\n"
            + fx_txt +
            f"- En el **Stress de febrero** los activos de alta volatilidad fueron los más castigados.")
        st.markdown("**Llega un shock triple (17-ago-2026):**")
        st.markdown(
            f"- **TW_01** · Tasa a 10 años **+{rate_bp} pb**: **{shock['tasa_sigmas'] * rate_bp / E.BASE_RATE_BP:.0f} σ** "
            f"semanales (la mayor alza observada fue {shock['tasa_max_semanal_bp']:.1f} pb).\n"
            f"- **TW_02** · CL_02 debe disponer del **30%** del portafolio en 6 meses.\n"
            f"- **TW_03** · COP **-{fx_pct if rules['fx_shock_applies'] else 8}%**: "
            f"**{shock['cop_sigmas']:.0f} σ** de 5 días (máximo observado: {pct(shock['cop_max_5d'])}).")

    # ---------- 2. ¿Qué datos o señales lo explican? ----------
    st.header("2 · ¿Qué datos o señales lo explican?")
    c1, c2 = st.columns(2)
    ret_loc = (1 + R).prod() - 1
    vol_loc = R.std() * np.sqrt(252)
    with c1:
        fig = go.Figure()
        for a in E.ASSETS:
            fig.add_trace(go.Scatter(
                x=[vol_loc[a]], y=[ret_loc[a]], mode="markers+text", text=[a[3:]], textposition="top center",
                marker=dict(size=14, color=COLOR[a], line=dict(color="white", width=2)), name=label(a, ref),
                hovertemplate=f"<b>{label(a, ref)}</b><br>Volatilidad %{{x:.1%}}<br>Retorno %{{y:.1%}}<extra></extra>"))
        fig.add_vline(x=E.HIGH_VOL, line=dict(color=NEUTRAL, dash="dot"),
                      annotation_text="alta volatilidad (>30%)", annotation_font=dict(size=11, color=INK_2))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        fig.update_xaxes(title=f"Volatilidad anual ({cur})", tickformat=".0%")
        fig.update_yaxes(title=f"Retorno 12 meses ({cur})", tickformat=".0%")
        st.plotly_chart(style(fig, 360, legend=False).update_layout(title="Riesgo y retorno: arriba a la izquierda es mejor"),
                        width="stretch")
    with c2:
        margin = E.last_quarter_margin(d["fund"], ref)
        r12_usd = close.iloc[-1] / close.iloc[0] - 1
        corr = margin.corr(r12_usd)
        fig = go.Figure()
        for a in E.ASSETS:
            fig.add_trace(go.Scatter(
                x=[margin[a]], y=[r12_usd[a]], mode="markers+text", text=[a[3:]], textposition="top center",
                marker=dict(size=14, color=COLOR[a], line=dict(color="white", width=2)), name=label(a, ref),
                hovertemplate=f"<b>{label(a, ref)}</b><br>Margen EBITDA %{{x:.1%}}<br>Retorno %{{y:.1%}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        fig.update_xaxes(title="Margen EBITDA del último trimestre", tickformat=".0%")
        fig.update_yaxes(title="Retorno 12 meses (USD)", tickformat=".0%")
        st.plotly_chart(style(fig, 360, legend=False).update_layout(
            title=f"Las empresas más rentables son las que suben (correlación {corr:.2f})".replace(".", ",")),
            width="stretch")
    with st.expander("Eventos y reacción del precio"):
        ev = E.event_reactions(close, d["events"], ref)
        st.dataframe(ev, hide_index=True, width="stretch", column_config={
            "5 días": st.column_config.NumberColumn(format="percent"),
            "20 días": st.column_config.NumberColumn(format="percent")})
        st.caption("Los eventos negativos de severidad alta se tradujeron en caídas de 10% a 15% en las semanas siguientes.")

    # ---------- 3. ¿Qué significa para el cliente? ----------
    st.header(f"3 · ¿Qué significa para {cid}?")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        dd_loc = (1 + R).cumprod().apply(E.max_drawdown)
        vol_usd = r_usd.std() * np.sqrt(252)
        tbl = pd.DataFrame({
            "Activo": [label(a, ref) for a in E.ASSETS],
            "País": ref.loc[E.ASSETS, "country"].values,
            f"Retorno 12m ({cur})": ret_loc.values,
            f"Caída máx. ({cur})": dd_loc.values,
            "Cumple solo el límite": np.where(np.abs(dd_loc.values) <= lim, "✅", "—"),
            "Alta volatilidad": np.where(vol_usd.values > E.HIGH_VOL, "Sí", "No"),
            "Sensibilidad a tasas": E.RATE_SCALE.reindex(E.ASSETS).values,
            "Retorno esperado prudente": mu.values,
            "Peso recomendado": w.values,
        })
        st.dataframe(tbl, hide_index=True, width="stretch", height=330, column_config={
            f"Retorno 12m ({cur})": st.column_config.NumberColumn(format="percent"),
            f"Caída máx. ({cur})": st.column_config.NumberColumn(format="percent"),
            "Retorno esperado prudente": st.column_config.NumberColumn(
                format="percent", help="50% retorno propio + 50% promedio + shock de tasas por sector (04 §1)"),
            "Sensibilidad a tasas": st.column_config.NumberColumn(help="-2 muy perjudicado … +1 beneficiado (juicio por sector)"),
            "Peso recomendado": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=0.4),
        })
    with c2:
        n_ok = int((np.abs(dd_loc) <= lim).sum())
        with st.container(border=True):
            st.markdown(f"**{clients.at[cid, 'priority']}**")
            st.markdown(f"- Tolerancia al riesgo: **{clients.at[cid, 'risk_tolerance']}**\n"
                        f"- Necesidad de liquidez: **{clients.at[cid, 'liquidity_need']}**"
                        + (" → **30% en 6 meses (TW_02)**" if cid == "CL_02" else "") + "\n"
                        f"- Regla: *{clients.at[cid, 'constraint']}*")
        st.metric("Activos que cumplen solos su límite", f"{n_ok} de 8", border=True)
        st.metric("Carteras posibles que cumplen todas sus reglas", pct(feas_ok / feas_all, 0),
                  f"{feas_ok:,} de {feas_all:,}".replace(",", "."), delta_color="off", border=True)
        st.metric("Probabilidad de superar el límite con una cartera genérica", pct(p_breach_ew), border=True)

    # ---------- 4. ¿Qué debería hacer? ----------
    st.header("4 · ¿Qué debería hacer?")
    c1, c2, c3 = st.columns([1, 1.4, 1.2])
    with c1:
        alloc = w[w > 0].sort_values()
        names = [label(a, ref) for a in alloc.index]
        colors = [COLOR[a] for a in alloc.index]
        if cash > 0:
            alloc = pd.concat([pd.Series({"CASH": cash}), alloc])
            names = [f"Liquidez en {cur}"] + names
            colors = [CASH_COLOR] + colors
        fig = go.Figure(go.Bar(x=alloc.values, y=names, orientation="h", marker=dict(color=colors, cornerradius=4),
                               text=[pct(v, 0) for v in alloc.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, alloc.max() * 1.3])
        st.plotly_chart(style(fig, 330, legend=False).update_layout(title="Cartera recomendada"), width="stretch")
    with c2:
        fig = go.Figure()
        for nm, mm, color, width in [("Partes iguales", m_ew, NEUTRAL, 1.5), ("Recomendada", m, PALETTE[0], 2.5)]:
            c = mm["curve"]
            fig.add_trace(go.Scatter(x=c.index, y=c / c.cummax() - 1, name=nm, line=dict(color=color, width=width),
                                     hovertemplate=nm + ": %{y:.1%}<extra></extra>"))
        fig.add_hline(y=-lim, line=dict(color="#0b0b0b", dash="dash", width=1.5),
                      annotation_text=f"Límite del cliente ({pct(-lim, 0)})", annotation_position="bottom left")
        fig.update_yaxes(tickformat=".0%", title=f"Caída desde el máximo ({cur})")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig, 330).update_layout(title="La cartera recomendada se mantiene dentro del límite"),
                        width="stretch")
    with c3:
        st.markdown("**Próximas acciones (Next Best Action)**")
        if cid == "CL_02":
            actions = [
                f"**Convertir hoy a COP el {pct(cash, 0)} de liquidez.** Protege **{pts(protected)}** del patrimonio "
                f"si el COP se recupera {rec_pct}% antes del retiro.",
                f"**Mantener el {pct(1 - cash, 0)} invertido en el núcleo** {core_txt}: caída máx. "
                f"{pct(m['caida_max'])} frente al límite de {pct(-lim, 0)}.",
                "**Limitar A08 a 10% y dejar A02 fuera:** son los más expuestos a la subida de tasas.",
                f"**No rebalancear por la tasa todavía:** el shock está a {shock['tasa_sigmas'] * rate_bp / E.BASE_RATE_BP:.0f} σ "
                "de la historia; revisar con los primeros precios después del twist.",
                "**Confirmar con el cliente** la fecha y el monto del retiro (TW_02).",
            ]
        else:
            actions = [
                f"**Mantener el núcleo** {core_txt} ({pct(w[core].sum(), 0)} del portafolio).",
                "**Limitar A02 y A08 a 10%**: los más expuestos a la subida de tasas.",
                f"**Vigilar el liderazgo de A04 y A05**: si caen más de {pct(lim * 0.7, 0)} desde su máximo, rotar a los siguientes del núcleo.",
                "**Revisar** con los primeros precios después del twist.",
            ]
        for i, a in enumerate(actions, 1):
            st.markdown(f"{i}. {a}")

    # ---------- 5. ¿Qué podría cambiar la recomendación? ----------
    st.header("5 · ¿Qué riesgo, supuesto o cambio podría modificar la recomendación?")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        for nm, dd, color in [("Partes iguales", dd_ew_b, NEUTRAL), ("Recomendada", dd_b, PALETTE[0])]:
            fig.add_trace(go.Histogram(x=dd, name=nm, marker_color=color, opacity=0.75, nbinsx=50,
                                       hovertemplate=nm + ": %{y} años con caída ≈ %{x:.0%}<extra></extra>"))
        fig.add_vline(x=-lim, line=dict(color="#0b0b0b", dash="dash", width=1.5),
                      annotation_text=f"Límite {pct(-lim, 0)}", annotation_position="top left")
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(tickformat=".0%", title=f"Peor caída del año simulado ({cur})")
        fig.update_yaxes(title="Años simulados")
        st.plotly_chart(style(fig, 360).update_layout(
            title=f"2.000 años posibles: supera el límite en {pct(p_breach)} de los casos (frente a {pct(p_breach_ew)})"),
            width="stretch")
        if nb_case:
            st.caption("Las probabilidades son las del notebook 04 (recommendation_robustness.csv); "
                       "el histograma es una simulación equivalente hecha por el dashboard.")
    with c2:
        rows = [("TW_01 · Solo tasa", rate_hit)]
        if rules["fx_shock_applies"]:
            rows += [
                ("TW_03 · Solo moneda", (1 - cash) * fx_shock),
                ("TW_01 + TW_03 combinado", twist),
                (f"Combinado y el COP se recupera {rec_pct}% (liquidez ya en COP)",
                 cash + (1 - cash + rate_hit) * (1 + fx_shock) * (1 - rec) - 1),
                (f"Combinado y el COP se recupera {rec_pct}% (liquidez en USD)",
                 (1 + rate_hit) * (1 + fx_shock) * (1 - rec) - 1),
            ]
        sc = pd.DataFrame(rows, columns=["Escenario", f"Impacto en {cur}"])
        sc["¿Dentro del límite?"] = np.where(sc[f"Impacto en {cur}"] > -lim, "✅", "❌")
        st.markdown("**Prueba de los escenarios del twist**")
        st.dataframe(sc, hide_index=True, width="stretch",
                     column_config={f"Impacto en {cur}": st.column_config.NumberColumn(format="percent")})
        st.markdown(
            "**Lo que podría cambiar la decisión**\n"
            f"- **Riesgo residual:** incluso esta cartera supera el límite en {pct(p_breach)} de los años simulados.\n"
            "- **Que A04 y A05 dejen de liderar:** el retorno esperado prudente depende de ellos.\n"
            "- **Que el shock de tasas pegue más de lo supuesto:** el efecto por sector es juicio experto, no una estimación.\n"
            "- **Supuestos:** liquidez al 0%, 271 días de historia y el archivo diario como fuente oficial de precios.")

# ---------- Calidad de datos ----------
with tab_quality:
    st.header("Find the truth: ¿qué tan confiables son los datos?")
    issues = E.load_issues()
    if issues is not None:
        k = st.columns(3)
        k[0].metric("Problemas distintos", len(issues), border=True)
        k[1].metric("Archivos con problemas", issues.tabla.nunique() - int("Entre archivos" in set(issues.tabla)),
                    "+ la contradicción entre diario e intradía" if "Entre archivos" in set(issues.tabla) else None,
                    delta_color="off", border=True)
        k[2].metric("Tablas sin problemas", "Activos, clientes y eventos", border=True)
        by_table = issues.groupby("tabla").size().sort_values()
        fig = go.Figure(go.Bar(x=by_table.values, y=by_table.index, orientation="h",
                               marker=dict(color=PALETTE[0], cornerradius=4),
                               text=by_table.values, textposition="outside",
                               hovertemplate="%{y}: %{x} problemas<extra></extra>"))
        fig.update_xaxes(range=[0, by_table.max() * 1.15], dtick=1)
        st.plotly_chart(style(fig, 300, legend=False).update_layout(title="Problemas por tabla de origen"),
                        width="stretch")
        tables = st.multiselect("Tabla", sorted(issues.tabla.unique()), default=sorted(issues.tabla.unique()))
        st.dataframe(issues[issues.tabla.isin(tables)], hide_index=True, width="stretch",
                     column_config={"filas": st.column_config.NumberColumn("Filas afectadas", format="%d")})
        st.caption("Fuente: data_clean/quality_issues.csv, generado por notebooks/01_calidad_y_limpieza.ipynb.")
    else:
        st.info("No se encontró data_clean/quality_issues.csv. Corre el notebook 01_calidad_y_limpieza.ipynb para generarlo.")

    st.subheader("¿Qué fuente de precios usar?")
    st.markdown("El archivo diario y el intradía no coinciden. La fuente confiable es la que se mueve con el "
                "**factor de mercado** de la tabla macro.")
    if st.toggle("Calcular la validación de fuentes (prices_expanded_daily.csv)"):
        exp = get_expanded().reindex(close.index)
        mf = macro.market_factor
        cmp = pd.DataFrame({"Diario": r_usd.corrwith(mf), "Intradía": exp.pct_change().corrwith(mf)})
        fig = go.Figure()
        for nm, color in [("Diario", PALETTE[0]), ("Intradía", PALETTE[1])]:
            fig.add_trace(go.Bar(x=[a[3:] for a in E.ASSETS], y=cmp[nm], name=f"{nm} (promedio {cmp[nm].mean():.2f})".replace(".", ","),
                                 marker=dict(color=color, cornerradius=4), hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.update_yaxes(title="Correlación con el factor de mercado")
        st.plotly_chart(style(fig, 320).update_layout(title="El archivo diario se mueve con el mercado; el intradía no"),
                        width="stretch")

# ---------- Metodología ----------
with tab_method:
    st.header("Cómo se construye la recomendación")
    st.markdown(
        "1. **Limpieza** (notebook 01): IDs unificados, fechas mixtas, volúmenes con `K`, precio ×100, duplicados y fundamentales corregidos.\n"
        "2. **Validación de la fuente** (notebook 02): el archivo diario es la fuente oficial porque concuerda con macro y eventos.\n"
        "3. **Twist** (notebook 03): los shocks están fuera de la historia, así que se modelan como escenarios.\n"
        "4. **Motor** (notebook 04): carteras aleatorias → reglas del cliente → riesgo en su moneda → objetivo según su prioridad → bootstrap.\n"
        "5. **Este dashboard**: lee `data_clean/` (tablas limpias del 01; `recommendations.csv` y `recommendation_robustness.csv` del 04) "
        "y recalcula en vivo las métricas y los escenarios del twist. Si se mueve la liquidez, también recalcula el bootstrap.")
    st.subheader("Supuestos")
    st.dataframe(pd.DataFrame([
        ("Retorno esperado", "50% retorno histórico del activo + 50% promedio de todos",
         "Con un año de datos el retorno histórico es casi todo ruido"),
        ("Shock de tasas", "Escala por sector (-2 a +1) × 3% por punto ante +65 pb",
         "La historia no permite estimar el efecto (t-stats < 2)"),
        ("Alta volatilidad", "Volatilidad anual > 30%", "Para medir la regla de CL_02"),
        ("Tope por activo", "35% (25% para CL_02)", "Regla prudencial; no viene del perfil"),
        ("Activos sensibles a tasas", "Máx. 10% en A02 y A08", "Los más expuestos a TW_01"),
        ("Liquidez de CL_02", "30% en COP, retorno 0%", "TW_02; no hay efectivo ni renta fija en el dataset"),
    ], columns=["Supuesto", "Valor", "Por qué"]), hide_index=True, width="stretch")
    st.subheader("Reglas y objetivo por cliente")
    st.dataframe(pd.DataFrame([
        {"Cliente": c, "Perfil": clients.at[c, "profile_name"], "Moneda": r["currency"],
         "Caída máx.": pct(r["dd_limit"], 0), "Tope por activo": pct(r["cap"], 0),
         "Liquidez": pct(r["cash"], 0), "Objetivo": r["objective"]} for c, r in RULES.items()]),
        hide_index=True, width="stretch")
    st.subheader("Sensibilidad de cada activo a tasas")
    st.dataframe(pd.DataFrame({"Activo": [label(a, ref) for a in E.ASSETS],
                               "Sensibilidad": E.RATE_SCALE.reindex(E.ASSETS).values,
                               "Razón": [E.RATE_REASON[a] for a in E.ASSETS]}), hide_index=True, width="stretch")
