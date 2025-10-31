# option.py
"""
Streamlit app: Illiquid option market simulation (Plotly-based)
Replaces matplotlib usage (fixes import error).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Illiquid Option Simulation (Plotly)", layout="wide")
st.title("📉 Illiquid Option Market Simulation — Plotly version")
st.write(
    "Educational simulation of an illiquid option where a reactive algo and a normal buyer interact. "
    "This version avoids matplotlib to prevent import issues in some Streamlit environments."
)

# ------------------ Sidebar controls ------------------
st.sidebar.header("Simulation controls")
fair_price = st.sidebar.number_input("Fair Value of Option", min_value=1.0, max_value=1000.0, value=40.0, step=1.0)
initial_bid = st.sidebar.number_input("Initial Bid (Algo Buyer)", min_value=0.01, max_value=1000.0, value=20.0, step=1.0)
initial_ask = st.sidebar.number_input("Initial Ask (Algo Seller)", min_value=0.01, max_value=5000.0, value=100.0, step=1.0)
human_limit_price = st.sidebar.number_input("Human Buy Limit Price", min_value=0.0, max_value=5000.0, value=21.0, step=0.5)
human_order_time = st.sidebar.slider("Human Order Time Step", min_value=0, max_value=200, value=12)
steps = st.sidebar.slider("Simulation Length (time steps)", min_value=10, max_value=500, value=60)
algo_reactive_buy_price = st.sidebar.number_input("Algo Reactive Buy Price", min_value=0.01, max_value=5000.0, value=22.0, step=0.5)
algo_sell_threshold_pct = st.sidebar.slider("Algo Sell Threshold (% above fair)", min_value=0, max_value=500, value=20)
push_strength = st.sidebar.slider("Algo Push Strength", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
min_spread = st.sidebar.number_input("Minimum Bid-Ask Spread", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
algo_trade_qty = st.sidebar.number_input("Algo trade qty per micro-step", min_value=1, max_value=1000, value=5, step=1)
human_qty = st.sidebar.number_input("Human order qty", min_value=1, max_value=10000, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Tweak parameters and observe how quotes & human P&L change.")

# ------------------ Simulation ------------------
np.random.seed(42)

algo_sell_threshold = fair_price * (1 + algo_sell_threshold_pct / 100.0)

bid = float(initial_bid)
ask = float(initial_ask)

time = []
bid_hist = []
ask_hist = []
mid_hist = []
pnl_hist = []

trades = []  # records as dicts: {'t', 'actor', 'action', 'price', 'qty'}

human_position = 0
human_avg_price = None
human_order_posted = False
human_order_filled = False

for t in range(steps):
    time.append(t)
    bid_hist.append(bid)
    ask_hist.append(ask)
    mid = (bid + ask) / 2.0
    mid_hist.append(mid)

    # Human posts a visible limit buy at the specified time
    if t == human_order_time:
        human_order_posted = True
        trades.append({"t": t, "actor": "human", "action": "post_limit_buy", "price": human_limit_price, "qty": human_qty})

    # If human posted and not filled, algo may react
    if human_order_posted and not human_order_filled:
        # If human limit is already >= ask, immediate fill at ask (human hits ask)
        if human_limit_price >= ask:
            exec_price = ask
            human_position += human_qty
            human_avg_price = exec_price if human_avg_price is None else (human_avg_price * (human_position - human_qty) + exec_price * human_qty) / human_position
            human_order_filled = True
            trades.append({"t": t, "actor": "human", "action": "filled_at_ask", "price": exec_price, "qty": human_qty})
        else:
            # Algo detects visible buy interest and pushes price via micro-buys
            for it in range(3):  # a few micro-iterations per time-step
                trades.append({"t": t, "actor": "algo", "action": "micro_buy", "price": algo_reactive_buy_price, "qty": int(algo_trade_qty)})
                # move bid/ask up to simulate consumed supply / aggressive buying
                bid = max(bid, algo_reactive_buy_price) + push_strength * 0.4
                ask = max(ask, bid + min_spread) + push_strength * 0.6
                # small mean reversion (market makers adjusting toward fair)
                bid += 0.03 * (fair_price - bid)
                ask += 0.03 * (fair_price - ask)
                mid = (bid + ask) / 2.0
                if mid >= algo_sell_threshold:
                    break

            # After algo push, if ask <= human limit, fill human
            if human_limit_price >= ask:
                exec_price = ask
                human_position += human_qty
                human_avg_price = exec_price if human_avg_price is None else (human_avg_price * (human_position - human_qty) + exec_price * human_qty) / human_position
                human_order_filled = True
                trades.append({"t": t, "actor": "human", "action": "filled_after_push", "price": exec_price, "qty": human_qty})

    # If mid crosses algo sell threshold and human hasn't filled, algo flips and sells to human
    mid = (bid + ask) / 2.0
    if mid >= algo_sell_threshold and human_order_posted and not human_order_filled:
        inflated_price = algo_sell_threshold  # algo sells at threshold
        trades.append({"t": t, "actor": "algo", "action": "sell_to_human_inflated", "price": inflated_price, "qty": human_qty})
        human_position += human_qty
        human_avg_price = inflated_price if human_avg_price is None else (human_avg_price * (human_position - human_qty) + inflated_price * human_qty) / human_position
        human_order_filled = True
        # reset wide quotes after flip
        bid = initial_bid
        ask = initial_ask
        trades.append({"t": t, "actor": "algo", "action": "reset_quotes", "price": None, "qty": None})

    # small random noise, enforce min spread
    bid += float(np.random.normal(0, 0.25))
    ask += float(np.random.normal(0, 0.25))
    if ask - bid < min_spread:
        ask = bid + float(min_spread)
    bid = max(0.01, bid)
    ask = max(bid + float(min_spread), ask)

    # record human P&L mark-to-fair
    if human_position > 0:
        pnl = (fair_price - human_avg_price) * human_position
    else:
        pnl = 0.0
    pnl_hist.append(pnl)

# ------------------ DataFrames ------------------
df_quotes = pd.DataFrame({
    "time": time,
    "bid": bid_hist,
    "ask": ask_hist,
    "mid": mid_hist,
    "human_pnl": pnl_hist
})

df_trades = pd.DataFrame(trades)

# ------------------ Plotly visualization ------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_quotes["time"], y=df_quotes["bid"], mode="lines", name="Bid"))
fig.add_trace(go.Scatter(x=df_quotes["time"], y=df_quotes["ask"], mode="lines", name="Ask"))
fig.add_trace(go.Scatter(x=df_quotes["time"], y=df_quotes["mid"], mode="lines", name="Mid (bid+ask)/2", line=dict(dash="dash")))

# horizontal lines for fair and algo-threshold
fig.add_hline(y=fair_price, line_dash="dot", annotation_text=f"Fair = {fair_price}", annotation_position="bottom left")
fig.add_hline(y=algo_sell_threshold, line_dash="dash", annotation_text=f"Algo sell thresh = {algo_sell_threshold:.2f}", annotation_position="top left")

# mark trade events
if not df_trades.empty:
    # plot human fills and algo sells specially
    for idx, r in df_trades.iterrows():
        t = r["t"]
        a = r["actor"]
        action = r["action"]
        price = r["price"]
        if price is None:
            continue
        marker = dict(size=10)
        if a == "human" and "filled" in action:
            fig.add_trace(go.Scatter(x=[t], y=[price], mode="markers", name=f"Human fill @ {price:.2f}", marker=dict(symbol="x", size=12)))
        elif a == "algo" and action.startswith("sell_to_human"):
            fig.add_trace(go.Scatter(x=[t], y=[price], mode="markers", name=f"Algo sold @ {price:.2f}", marker=dict(symbol="diamond", size=12)))
        elif a == "algo" and action == "micro_buy":
            # skip plotting every micro buy to avoid clutter
            pass

fig.update_layout(
    title="Quote Evolution (Bid / Ask / Mid)",
    xaxis_title="Time step",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# P&L subplot
fig_pnl = go.Figure()
fig_pnl.add_trace(go.Scatter(x=df_quotes["time"], y=df_quotes["human_pnl"], mode="lines+markers", name="Human unrealized P&L"))
fig_pnl.update_layout(title="Human mark-to-fair unrealized P&L", xaxis_title="Time step", yaxis_title="P&L")

# show side-by-side
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.plotly_chart(fig_pnl, use_container_width=True)

# ------------------ Data tables & summary ------------------
st.subheader("Trade / Event Log")
if df_trades.empty:
    st.write("No trades recorded in the simulation.")
else:
    # format price column
    df_trades_display = df_trades.copy()
    df_trades_display["price"] = df_trades_display["price"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    st.dataframe(df_trades_display)

st.subheader("Price & P&L (sample rows)")
st.dataframe(df_quotes.head(200))

st.markdown("---")
if human_position > 0:
    st.success(f"Final human position: {human_position} units at avg price {human_avg_price:.2f}")
    st.info(f"Mark-to-fair P&L: {pnl_hist[-1]:.2f}")
else:
    st.info("Human had no position (no fills).")

st.caption("This simulation is illustrative and educational only — do not use for live trading.")
