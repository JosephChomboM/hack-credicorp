"""Datos y motor de recomendación del dashboard.

Todos los resultados salen de data_clean/:
- tablas limpias que genera notebooks/01_calidad_y_limpieza.ipynb,
- carteras y robustez que exporta notebooks/04_find_the_value.ipynb.
Lo único que vive en este archivo son los supuestos y reglas de optimización del 04 (§1 y §4), que el notebook no exporta.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
TWIST_FILE = ROOT / "dataset_twist.csv"

ASSETS = [f"VF_A0{i}" for i in range(1, 9)]

# ---------- Supuestos del motor (04 §1) ----------
SHRINK = 0.5                 # peso del retorno histórico propio vs. el promedio
RATE_PCT_PER_POINT = 0.03    # cada punto de sensibilidad = 3% de precio ante +65 pb
BASE_RATE_BP = 65
HIGH_VOL = 0.30
TWIST_CAP = 0.10             # máx. por activo muy sensible a tasas
RATE_SCALE = pd.Series({"VF_A01": -1, "VF_A02": -2, "VF_A03": 0, "VF_A04": 1,
                        "VF_A05": -1, "VF_A06": -1, "VF_A07": 0, "VF_A08": -2})
RATE_REASON = {
    "VF_A01": "Consumo defensivo: cae poco si el mercado se ajusta",
    "VF_A02": "Tecnología de crecimiento: muy sensible a la tasa de descuento",
    "VF_A03": "Energía: suele resistir alzas de tasa por inflación",
    "VF_A04": "Financiero: más margen de interés con tasas altas",
    "VF_A05": "Salud: defensivo, sensibilidad moderada",
    "VF_A06": "Industrial: cíclico y con la mayor deuda relativa",
    "VF_A07": "Materiales: ligado a inflación, demanda externa débil",
    "VF_A08": "Utilities: se comporta como un bono; pierde con tasas altas",
}

# ---------- Reglas de optimización por cliente que el 04 no exporta (04 §4) ----------
# La moneda y la caída máxima salen de client_profiles.csv; la liquidez, de recommendations.csv.
RULES_04 = {
    "CL_01": dict(cap=0.35, objective="Maximizar retorno esperado", fx_shock_applies=False),
    "CL_02": dict(cap=0.25, high_vol_cap=0.25, min_return=0.0, n_samples=300000, fx_shock_applies=True,
                  objective="Minimizar pérdidas extremas (CVaR) con retorno esperado ≥ 0"),
    "CL_03": dict(cap=0.35, min_div=3, fx_shock_applies=False, objective="Mejor retorno por unidad de riesgo (Sharpe)"),
}


# ---------- Carga desde data_clean ----------
def _read(name, **kw):
    return pd.read_csv(CLEAN / name, encoding="utf-8-sig", **kw)


def load_data():
    prices = _read("prices_daily.csv", parse_dates=["date"])
    macro = _read("macro.csv", parse_dates=["date"]).set_index("date")
    fund = _read("fundamentals.csv", parse_dates=["period_end"])
    ref = _read("asset_reference.csv").set_index("asset_id")
    clients = _read("client_profiles.csv").set_index("client_id")
    events = _read("events.csv", parse_dates=["event_date"])
    twist = pd.read_csv(TWIST_FILE, encoding="utf-8-sig", parse_dates=["effective_date"])

    # Carteras del 04: filas VF_A0x = activos; el resto (LIQUIDEZ_COP, …) = liquidez
    recs = _read("recommendations.csv")
    is_asset = recs.asset_id.isin(ASSETS)
    weights = (recs[is_asset].pivot(index="asset_id", columns="client_id", values="weight_pct")
               .reindex(ASSETS).fillna(0) / 100)
    cash = recs[~is_asset].groupby("client_id").weight_pct.sum().reindex(clients.index).fillna(0) / 100
    rules = {c: {**RULES_04[c], "currency": clients.at[c, "base_currency"],
                 "dd_limit": clients.at[c, "max_drawdown_tolerance_pct"] / 100, "cash": float(cash[c])}
             for c in weights.columns}

    close = prices.pivot(index="date", columns="asset_id", values="close")[ASSETS]
    return dict(prices=prices, close=close, r_usd=close.pct_change().dropna(), macro=macro, fund=fund,
                ref=ref, clients=clients, events=events, twist=twist, weights=weights, rules=rules,
                metrics=_read("recommendation_metrics.csv").set_index("client_id"),
                robustness=_read("recommendation_robustness.csv").set_index("client_id"))


def load_expanded_daily():
    """Cierre diario del archivo intradía (notebook 01); solo para validar fuentes."""
    e = _read("prices_expanded_daily.csv", parse_dates=["date"])
    return e.pivot(index="date", columns="asset_id", values="close_exp")[ASSETS]


def load_issues():
    path = CLEAN / "quality_issues.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


# ---------- Métricas ----------
def local_returns(r_usd, macro, currency):
    if currency == "USD":
        return r_usd
    fx = macro[f"usd_{currency.lower()}"].pct_change().reindex(r_usd.index)
    return (1 + r_usd).mul(1 + fx, axis=0) - 1


def max_drawdown(curve):
    return (curve / curve.cummax() - 1).min()


def expected_returns(close, pct_per_point=RATE_PCT_PER_POINT):
    hist = close.iloc[-1] / close.iloc[0] - 1
    return SHRINK * hist + (1 - SHRINK) * hist.mean() + RATE_SCALE * pct_per_point


def portfolio_weights(weights, client_id, cash):
    """Pesos de recommendations.csv reescalados a la liquidez elegida (la liquidez rinde 0%)."""
    w = weights[client_id]
    return w / w.sum() * (1 - cash)


def portfolio_metrics(R, w, mu):
    pr = R[ASSETS] @ w.values
    curve = (1 + pr).cumprod()
    q = pr.quantile(0.05)
    return {
        "caida_max": max_drawdown(curve),
        "vol": pr.std() * np.sqrt(252),
        "cvar95": pr[pr <= q].mean(),
        "peor_6m": (curve.iloc[126:].values / curve.iloc[:-126].values - 1).min(),
        "retorno_12m": curve.iloc[-1] - 1,
        "retorno_esperado": float(w @ mu),
        "curve": curve,
    }


def block_bootstrap(R, w, n_paths=2000, horizon=252, block=20, seed=7):
    """04 §8: 'años posibles' armados con bloques de 20 días de la historia."""
    rng = np.random.default_rng(seed)
    pr = (R[ASSETS] @ w.values).values
    starts = rng.integers(0, len(pr) - block, size=(n_paths, horizon // block + 1))
    sims = np.stack([np.concatenate([pr[s:s + block] for s in row])[:horizon] for row in starts])
    curve = np.cumprod(1 + sims, axis=1)
    dd = (curve / np.maximum.accumulate(curve, axis=1) - 1).min(axis=1)
    return dd, curve[:, -1] - 1


def twist_impact(w, cash, rate_pct_per_point, fx_shock):
    """04 §5: shock de tasas por sector sobre lo invertido y, si aplica, el FX sobre la parte en USD.
    La liquidez está en la moneda del cliente y no se mueve."""
    rate_hit = float((w * RATE_SCALE * rate_pct_per_point).sum())
    invested = 1 - cash
    if invested <= 0:
        return 0.0
    return invested * ((1 + rate_hit / invested) * (1 + fx_shock) - 1)


def feasibility(R, mu, rules, cash, ref, seed=7):
    """04 §3-4: de las carteras aleatorias que respetan las reglas de pesos, cuántas respetan también el límite de caída."""
    rng = np.random.default_rng(seed)
    n = rules.get("n_samples", 90000)
    W = np.vstack([rng.dirichlet(np.full(8, a), n // 3) for a in (1.0, 0.4, 0.15)]) * (1 - cash)
    vol = R[ASSETS].std() * np.sqrt(252)
    ok = (W <= rules["cap"] + 1e-9).all(axis=1)
    sens = (RATE_SCALE.reindex(ASSETS) <= -2).values
    ok &= (W[:, sens] <= TWIST_CAP + 1e-9).all(axis=1)
    if rules.get("high_vol_cap") is not None:
        ok &= W[:, (vol > HIGH_VOL).values].sum(axis=1) <= rules["high_vol_cap"] + 1e-9
    if rules.get("min_div"):
        country, sector = ref.loc[ASSETS, "country"].values, ref.loc[ASSETS, "sector"].values
        k = rules["min_div"]
        ok &= np.array([len(set(country[w >= .10])) >= k and len(set(sector[w >= .10])) >= k for w in W])
    W = W[ok]
    pr = R[ASSETS].values @ W.T
    curve = np.cumprod(1 + pr, axis=0)
    dd = (curve / np.maximum.accumulate(curve, axis=0) - 1).min(axis=0)
    feas = np.abs(dd) <= rules["dd_limit"]
    if rules.get("min_return") is not None:
        feas &= W @ mu.values >= rules["min_return"]
    return int(feas.sum()), len(W)


def shock_size(macro, fx_shock):
    y10 = macro.us_10y_yield
    wk_bp = y10.resample("W-FRI").last().diff() * 100
    dcop = macro.usd_cop.pct_change()
    return {
        "tasa_sigmas": BASE_RATE_BP / wk_bp.std(),
        "tasa_max_semanal_bp": wk_bp.max(),
        "cop_sigmas": fx_shock / (dcop.std() * np.sqrt(5)) if fx_shock else 0.0,
        "cop_max_5d": macro.usd_cop.pct_change(5).max(),
    }


def event_reactions(close, events, ref):
    sector_asset = dict(zip(ref.sector, ref.index))
    rows = []
    for _, ev in events.iterrows():
        a = ev.scope if ev.scope in ASSETS else sector_asset.get(ev.scope)
        i = close.index.searchsorted(ev.event_date)
        base = close[a].iloc[i - 1]
        rows.append({
            "Fecha": ev.event_date.date(), "Activo": f"{a[3:]} {ref.at[a, 'name']}", "Tipo": ev.event_type,
            "Severidad": ev.severity, "Titular": ev.headline,
            "5 días": close[a].iloc[min(i + 4, len(close) - 1)] / base - 1,
            "20 días": close[a].iloc[min(i + 19, len(close) - 1)] / base - 1,
        })
    return pd.DataFrame(rows)


def last_quarter_margin(fund, ref):
    last = fund.groupby("issuer_id").tail(1).set_index("issuer_id")
    m = (last.ebitda_usd_m / last.revenue_usd_m).rename("margen_ebitda")
    return m.rename(index={v: k for k, v in ref.issuer_id.items()}).reindex(ASSETS)
