"""
DSI433 Group 5 — Portfolio Optimization & Solvency (Interactive Demo)
ใช้ข้อมูลและสูตรชุดเดียวกับ notebook (v1 / v2 / v3)
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="DSI433 G5 · Portfolio & Solvency", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# 1. ข้อมูลบริษัท (Mock-up) — ค่าเดียวกับ insurance_mockup_data.py
# ------------------------------------------------------------------
TOTAL_ASSETS = 100_000_000_000
LIABILITIES = 80_000_000_000
EQUITY = 20_000_000_000
CURRENT_CAR = 2.50
LIAB_DURATION = 10.0
FUNDING_RATIO = LIABILITIES / TOTAL_ASSETS

ASSETS = ["Cash_THOR", "Gov_Bond", "Corp_Bond", "Thai_Equity_SET", "Global_Equity_ACWI"]
LABELS = {"Cash_THOR": "Cash (THOR)", "Gov_Bond": "Gov Bond", "Corp_Bond": "Corp Bond",
          "Thai_Equity_SET": "Thai Equity (SET)", "Global_Equity_ACWI": "Global Equity (ACWI)"}
W_CURRENT = np.array([0.05, 0.60, 0.20, 0.10, 0.05])
LIQUIDITY = np.array([1.00, 0.95, 0.60, 0.85, 0.80])
RISK_CHARGE = np.array([0.00, 0.00, 0.04, 0.25, 0.30])
MIN_LIMIT = np.array([0.02, 0.20, 0.00, 0.00, 0.00])
MAX_LIMIT = np.array([1.00, 1.00, 0.60, 0.30, 0.15])
IS_EQUITY = np.array([a in ("Thai_Equity_SET", "Global_Equity_ACWI") for a in ASSETS])
IS_HIGH_LIQUID = LIQUIDITY >= 0.95
SET_IDX = ASSETS.index("Thai_Equity_SET")

MODEL_INFO = {
    "v1 · Max Sharpe": "หาสัดส่วนที่ Sharpe Ratio สูงสุด (มองแค่สินทรัพย์)",
    "v2 · Max Sharpe + ALM": "เหมือน v1 + บังคับช่องว่าง Duration กับหนี้สินไม่เกินค่าที่กำหนด",
    "v3 · Min Surplus Vol": "หาสัดส่วนที่ความผันผวนของส่วนทุน (สินทรัพย์ − หนี้สิน) ต่ำสุด",
}
COLORS = {"Current": "#8a8984", "v1 · Max Sharpe": "#eb6834",
          "v2 · Max Sharpe + ALM": "#1baf7a", "v3 · Min Surplus Vol": "#6250d6"}


# ------------------------------------------------------------------
# 2. โหลดข้อมูลตลาดจริง และคำนวณพารามิเตอร์ (เหมือน notebook)
# ------------------------------------------------------------------
@st.cache_data
def load_parameters(path: str):
    df = pd.read_csv(path, index_col="Month")
    returns = df[ASSETS].dropna()

    mu = returns.mean().values * 12
    sigma = returns.cov().values * 12
    risk_free = mu[0]

    latest = df[["GOV_Duration", "CORP_Duration"]].dropna().iloc[-1]
    duration = np.array([0.0, latest["GOV_Duration"], latest["CORP_Duration"], 0.0, 0.0])

    gov_yield = df["GOV_Yield"] / 100
    liab = (gov_yield.shift(1) / 12 - LIAB_DURATION * gov_yield.diff()).reindex(returns.index)
    mu_l = liab.mean() * 12
    var_l = liab.var() * 12
    cov_al = returns.apply(lambda c: c.cov(liab)).values * 12

    return dict(mu=mu, sigma=sigma, rf=risk_free, dur=duration, mu_l=mu_l, var_l=var_l,
                cov_al=cov_al, n_months=len(returns),
                period=f"{returns.index.min()} – {returns.index.max()}")


DATA_FILE = Path(__file__).parent / "app_data.csv"
if not DATA_FILE.exists():
    st.error("ไม่พบไฟล์ app_data.csv — ส่งออกจาก notebook แล้ววางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()
P = load_parameters(str(DATA_FILE))

OTHER_RISK_CAPITAL = EQUITY / CURRENT_CAR - float(np.sum(W_CURRENT * TOTAL_ASSETS * RISK_CHARGE))


# ------------------------------------------------------------------
# 3. ฟังก์ชันวัดผลพอร์ต
# ------------------------------------------------------------------
def port_return(w):     return float(w @ P["mu"])
def port_vol(w):        return float(np.sqrt(w @ P["sigma"] @ w))
def port_sharpe(w):     return (port_return(w) - P["rf"]) / port_vol(w)
def port_car(w):        return EQUITY / (float(np.sum(w * TOTAL_ASSETS * RISK_CHARGE)) + OTHER_RISK_CAPITAL)
def port_duration(w):   return float(w @ P["dur"])
def equity_ratio(w):    return float(w[IS_EQUITY].sum())
def liquid_ratio(w):    return float(w[IS_HIGH_LIQUID].sum())
def surplus_vol(w):
    var = w @ P["sigma"] @ w - 2 * FUNDING_RATIO * (w @ P["cov_al"]) + FUNDING_RATIO**2 * P["var_l"]
    return float(np.sqrt(max(var, 0.0)))
def rate_down_impact(w):
    return (port_duration(w) * TOTAL_ASSETS - LIAB_DURATION * LIABILITIES) * 0.01


# ------------------------------------------------------------------
# 4. Optimization (SLSQP, ลองหลายจุดเริ่มต้น)
# ------------------------------------------------------------------
@st.cache_data
def solve(model, car_target, max_equity, min_liquid, return_target, min_set, max_gap):
    lower, upper = MIN_LIMIT.copy(), MAX_LIMIT.copy()
    lower[SET_IDX] = max(lower[SET_IDX], min_set)
    bounds = list(zip(lower, upper))

    def constraints(with_return):
        cons = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: max_equity - equity_ratio(w)},
            {"type": "ineq", "fun": lambda w: port_car(w) - car_target},
            {"type": "ineq", "fun": lambda w: liquid_ratio(w) - min_liquid},
        ]
        if model.startswith("v2"):
            cons.append({"type": "ineq", "fun": lambda w: port_duration(w) - (LIAB_DURATION - max_gap)})
        if with_return:
            cons.append({"type": "ineq", "fun": lambda w: port_return(w) - return_target})
        return cons

    starts = [lower + (1 - lower.sum()) / len(lower), W_CURRENT, np.full(5, 0.2)]
    starts = [np.clip(w, lower, upper) / np.clip(w, lower, upper).sum() for w in starts]

    def run(objective, with_return):
        results = [minimize(objective, w0, method="SLSQP", bounds=bounds,
                            constraints=constraints(with_return),
                            options={"maxiter": 1000, "ftol": 1e-12}) for w0 in starts]
        ok = [r for r in results if r.success]
        return min(ok or results, key=lambda r: r.fun)

    max_ret = port_return(run(lambda w: -port_return(w), False).x)
    feasible = max_ret >= return_target - 1e-9
    objective = surplus_vol if model.startswith("v3") else (lambda w: -port_sharpe(w))
    res = run(objective, feasible)
    return res.x, feasible, bool(res.success), max_ret


# ------------------------------------------------------------------
# 5. Sidebar — ปรับค่า
# ------------------------------------------------------------------
st.sidebar.header("⚙️ ปรับค่าโมเดล")
model = st.sidebar.radio("เลือกโมเดล", list(MODEL_INFO), index=1)
st.sidebar.caption(MODEL_INFO[model])

st.sidebar.subheader("เกณฑ์บริษัทประกัน")
car_target = st.sidebar.slider("CAR เป้าหมาย (%)", 140, 300, 200, 10) / 100
max_equity = st.sidebar.slider("หุ้นรวมไม่เกิน (%)", 10, 40, 30, 5) / 100
min_liquid = st.sidebar.slider("เงินสด + พันธบัตรรัฐบาล ขั้นต่ำ (%)", 5, 50, 15, 5) / 100
return_target = st.sidebar.slider("ผลตอบแทนขั้นต่ำ = ต้นทุนเงินสำรอง (%)", 0.0, 5.0, 3.5, 0.1) / 100

st.sidebar.subheader("ทดลองเพิ่มเติม")
min_set = st.sidebar.slider("หุ้นไทยขั้นต่ำ (%)", 0, 15, 0, 1) / 100
max_gap = st.sidebar.slider("ช่องว่าง Duration สูงสุด (ปี) — ใช้กับ v2", 2.0, 8.0, 4.0, 0.5,
                            disabled=not model.startswith("v2"))

w_opt, feasible, ok, max_ret = solve(model, car_target, max_equity, min_liquid,
                                     return_target, min_set, max_gap)

# ------------------------------------------------------------------
# 6. หน้าหลัก
# ------------------------------------------------------------------
st.title("Portfolio Optimization for a Thai Life Insurer")
st.caption(f"DSI433 Group 5 · ข้อมูลตลาดจริง {P['period']} ({P['n_months']} เดือน) · "
           f"บริษัทสมมติ สินทรัพย์ 100,000 ล้านบาท")

if not ok:
    st.error("SLSQP หาคำตอบที่ผ่านทุกเงื่อนไขไม่ได้ — ลองผ่อนเกณฑ์ใน sidebar")
if not feasible:
    st.warning(f"⚠️ ภายใต้เกณฑ์ชุดนี้ ผลตอบแทนสูงสุดที่ทำได้คือ {max_ret:.2%} ต่ำกว่าเป้า "
               f"{return_target:.2%} — แสดงผลโดยตัดเกณฑ์ผลตอบแทนออก")

# ---- ตัวเลขสำคัญ ----
st.subheader(f"ผลลัพธ์: {model}")
cols = st.columns(4)
cols[0].metric("ผลตอบแทน / ปี", f"{port_return(w_opt):.2%}",
               f"{(port_return(w_opt) - port_return(W_CURRENT)) * 100:+.2f} pp vs Current")
cols[1].metric("Sharpe Ratio", f"{port_sharpe(w_opt):.2f}",
               f"{port_sharpe(w_opt) - port_sharpe(W_CURRENT):+.2f}")
cols[2].metric("CAR", f"{port_car(w_opt):.0%}",
               f"{(port_car(w_opt) - port_car(W_CURRENT)) * 100:+.0f} pp")
cols[3].metric("Surplus Volatility", f"{surplus_vol(w_opt):.2%}",
               f"{(surplus_vol(w_opt) - surplus_vol(W_CURRENT)) * 100:+.2f} pp", delta_color="inverse")

cols = st.columns(4)
cols[0].metric("ความเสี่ยงสินทรัพย์", f"{port_vol(w_opt):.2%}",
               f"{(port_vol(w_opt) - port_vol(W_CURRENT)) * 100:+.2f} pp", delta_color="inverse")
cols[1].metric("Duration Gap", f"{port_duration(w_opt) - LIAB_DURATION:.2f} ปี",
               f"{port_duration(w_opt) - port_duration(W_CURRENT):+.2f}")
cols[2].metric("ดอกเบี้ยลด 1% → ส่วนทุน", f"{rate_down_impact(w_opt) / 1e6:,.0f} ล้าน",
               f"{(rate_down_impact(w_opt) - rate_down_impact(W_CURRENT)) / 1e6:+,.0f} ล้าน")
cols[3].metric("สภาพคล่องสูง", f"{liquid_ratio(w_opt):.0%}",
               f"{(liquid_ratio(w_opt) - liquid_ratio(W_CURRENT)) * 100:+.0f} pp")

# ---- กราฟสัดส่วน ----
st.subheader("สัดส่วนการลงทุน: Current เทียบกับพอร์ตที่ได้")
chart_df = pd.DataFrame({
    "Asset": [LABELS[a] for a in ASSETS] * 2,
    "Portfolio": ["Current"] * 5 + [model] * 5,
    "Weight": np.concatenate([W_CURRENT, w_opt]),
})
chart_df["THB_mn"] = chart_df["Weight"] * TOTAL_ASSETS / 1e6
bars = (alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Asset:N", sort=[LABELS[a] for a in ASSETS], title=None),
            yOffset=alt.YOffset("Portfolio:N", sort=["Current", model]),
            x=alt.X("Weight:Q", axis=alt.Axis(format="%", grid=True), title="สัดส่วน (% ของสินทรัพย์)"),
            color=alt.Color("Portfolio:N", sort=["Current", model],
                            scale=alt.Scale(domain=["Current", model], range=[COLORS["Current"], COLORS[model]]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Portfolio:N"), alt.Tooltip("Asset:N"),
                     alt.Tooltip("Weight:Q", format=".1%", title="สัดส่วน"),
                     alt.Tooltip("THB_mn:Q", format=",.0f", title="ล้านบาท")])
        .properties(height=320))
st.altair_chart(bars, width='stretch')

# ---- ตาราง ----
left, right = st.columns([3, 2])
with left:
    st.markdown("**สัดส่วนและการโยกย้ายเงิน**")
    table = pd.DataFrame({
        "Current": W_CURRENT, model: w_opt,
        "ย้ายเงิน (ล้านบาท)": (w_opt - W_CURRENT) * TOTAL_ASSETS / 1e6,
    }, index=[LABELS[a] for a in ASSETS])
    st.dataframe(table.style.format({"Current": "{:.1%}", model: "{:.1%}", "ย้ายเงิน (ล้านบาท)": "{:+,.0f}"}),
                 width='stretch')
with right:
    st.markdown("**ตรวจเกณฑ์**")
    checks = {
        f"ผลตอบแทน ≥ {return_target:.1%}": port_return(w_opt) >= return_target - 1e-6,
        f"CAR ≥ {car_target:.0%}": port_car(w_opt) >= car_target - 1e-6,
        f"หุ้นรวม ≤ {max_equity:.0%}": equity_ratio(w_opt) <= max_equity + 1e-6,
        f"สภาพคล่องสูง ≥ {min_liquid:.0%}": liquid_ratio(w_opt) >= min_liquid - 1e-6,
        f"หุ้นไทย ≥ {min_set:.0%}": w_opt[SET_IDX] >= min_set - 1e-6,
    }
    if model.startswith("v2"):
        checks[f"Duration gap ≤ {max_gap:.1f} ปี"] = port_duration(w_opt) >= LIAB_DURATION - max_gap - 1e-6
    st.dataframe(pd.DataFrame({"ผล": ["✅ ผ่าน" if v else "❌ ไม่ผ่าน" for v in checks.values()]},
                              index=list(checks)), width='stretch')

with st.expander("ℹ️ สมมติฐานของโมเดล"):
    st.markdown(f"""
- **Optimizer:** `scipy.optimize.minimize` แบบ SLSQP (ลอง 3 จุดเริ่มต้น เลือกคำตอบที่ดีที่สุด)
- **Risk-free:** THOR เฉลี่ย {P['rf']:.2%} ต่อปี
- **CAR:** ส่วนทุน ÷ (Σ สัดส่วน × สินทรัพย์ × Risk Charge + ความเสี่ยงด้านอื่น {OTHER_RISK_CAPITAL/1e6:,.0f} ล้าน)
  — ความเสี่ยงด้านอื่นปรับเทียบให้พอร์ตปัจจุบันได้ CAR 250%
- **Risk Charge (Mock-up):** เงินสด 0%, พันธบัตร 0%, หุ้นกู้ 4%, หุ้นไทย 25%, หุ้นโลก 30%
- **หนี้สิน (v3):** ผลตอบแทนประมาณจาก Yield พันธบัตรรัฐบาล × Duration หนี้สิน 10 ปี
- **ข้อจำกัด:** ข้อมูลอดีต {P['n_months']} เดือน อาจไม่สะท้อนอนาคต · ข้อมูลบริษัทเป็นค่าสมมติ
""")
