import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import date
import pandas as pd

st.set_page_config(page_title="SPCX Options Pricer", layout="wide")

STRIKES = list(range(100, 210, 10))
EXPIRY  = date(2027, 1, 16)
SIGMA   = 1.0
R       = 0.04

def black_scholes(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

@st.cache_data(ttl=60)
def get_live_price():
    try:
        ticker = yf.Ticker("SPCX")
        hist = ticker.history(period="1d")
        if not hist.empty:
            price = round(float(hist["Close"].iloc[-1]), 2)
            if price > 0:
                return price, True
    except Exception:
        pass
    return 166.00, False

st.title("SPCX Option Pricer")
st.caption("Black-Scholes model · European calls and puts · Strikes $100–$200 · Expiry Jan 16, 2027")

price, is_live = get_live_price()

if is_live:
    st.success(f"Live price: **${price:.2f}** via Yahoo Finance — updates every 60 seconds")
else:
    st.warning(f"Using fallback price **$166.00** — Yahoo Finance unavailable or market closed")

S = price
today = date.today()
days_left = (EXPIRY - today).days
T = max(days_left, 1) / 365

c1, c2, c3, c4, c5, c6 = st.columns(6)
atm_k    = min(STRIKES, key=lambda x: abs(x - S))
atm_call = black_scholes(S, atm_k, T, R, SIGMA, "call")
atm_put  = black_scholes(S, atm_k, T, R, SIGMA, "put")

c1.metric("Stock price",    f"${S:.2f}")
c2.metric("Implied vol",    "100%")
c3.metric("Risk-free rate", "4.00%")
c4.metric("Days to expiry", days_left)
c5.metric("ATM Call",       f"${atm_call:.2f}")
c6.metric("ATM Put",        f"${atm_put:.2f}")

st.divider()

highlighted = st.select_slider(
    "Highlight a strike",
    options=STRIKES,
    value=atm_k,
    format_func=lambda x: f"${x}"
)

rows = []
for K in STRIKES:
    call  = black_scholes(S, K, T, R, SIGMA, "call")
    put   = black_scholes(S, K, T, R, SIGMA, "put")
    c_int = max(0.0, S - K)
    p_int = max(0.0, K - S)
    c_tv  = call - c_int
    p_tv  = put  - p_int

    if K == atm_k:
        status = "ATM"
    elif K < S:
        status = "Call ITM"
    else:
        status = "Put ITM"

    rows.append({
        "Strike":           f"${K}",
        "Status":           status,
        "Call price":       round(call,  2),
        "Call intrinsic":   round(c_int, 2),
        "Call time value":  round(c_tv,  2),
        "Put price":        round(put,   2),
        "Put intrinsic":    round(p_int, 2),
        "Put time value":   round(p_tv,  2),
        "_highlight":       K == highlighted
    })

df = pd.DataFrame(rows)

def style_row(row):
    K_val = int(row["Strike"].replace("$", ""))
    if K_val == highlighted:
        return ["background-color: rgba(91,142,245,0.15)"] * len(row)
    if row["Status"] == "ATM":
        return ["background-color: rgba(91,142,245,0.07)"] * len(row)
    if row["Status"] == "Call ITM":
        return ["background-color: rgba(45,200,122,0.05)"] * len(row)
    if row["Status"] == "Put ITM":
        return ["background-color: rgba(240,85,85,0.05)"] * len(row)
    return [""] * len(row)

display_df = df.drop(columns=["_highlight"])
styled = display_df.style.apply(style_row, axis=1)

st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()
with st.expander("How this works"):
    st.markdown("""
**Black-Scholes formula**

`Call = S·N(d₁) − K·e^(−rT)·N(d₂)`
`Put  = K·e^(−rT)·N(−d₂) − S·N(−d₁)`

Where `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)` and `d₂ = d₁ − σ√T`

**Inputs used**
- **S** = live SPCX stock price from Yahoo Finance (updates every 60 seconds)
- **K** = strike price ($100–$200 in $10 steps)
- **σ** = 100% implied volatility (realistic for a newly IPO'd stock)
- **r** = 4% risk-free rate (US Treasury proxy)
- **T** = days remaining to January 16, 2027, divided by 365

**Put-call parity check:** C − P = S − K·e^(−rT) holds for all rows.

**Intrinsic value** = immediate exercise value (how much the option is worth right now if exercised).
**Time value** = total price minus intrinsic value (what the market pays for the chance of future movement).
    """)

st.caption("Black-Scholes model · Yahoo Finance data · Educational purposes only · Not financial advice")
