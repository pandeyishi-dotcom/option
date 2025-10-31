import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Illiquid Option Simulation", layout="wide")

st.title("📉 Illiquid Option Market Simulation")
st.write("""
This simulation demonstrates how a low-liquidity option market with a reactive algorithm (algo)
can cause price distortions when a normal human buyer enters the market.
This is purely educational — **not for trading use**.
""")

# ---------------- USER CONTROLS ---------------- #
st.sidebar.header("Simulation Controls")

fair_price = st.sidebar.number_input("Fair Value of Option", 20.0, 100.0, 40.0, 1.0)
initial_bid = st.sidebar.number_input("Initial Bid (Algo Buyer)", 0.0, 100.0, 20.0, 1.0)
initial_ask = st.sidebar.number_input("Initial Ask (Algo Seller)", 0.0, 200.0, 100.0, 1.0)
human_limit_price = st.sidebar.number_input("Human Buy Limit Price", 0.0, 200.0, 21.0, 0.5)
human_order_time = st.sidebar.slider("Human Order Time Step", 0, 60, 12)
steps = st.sidebar.slider("Simulation Length (time steps)", 10, 200, 60)
algo_reactive_buy_price = st.sidebar.number_input("Algo Reactive Buy Price", 0.0, 200.0, 22.0, 0.5)
algo_sell_threshold_pct = st.sidebar.slider("Algo Sell Threshold (% above fair)", 0, 200, 20)
push_strength = st.sidebar.slider("Algo Push Strength", 0.1, 5.0, 1.0, 0.1)
min_spread = st.sidebar.number_input("Minimum Bid-Ask Spread", 0.5, 10.0, 1.0, 0.1)

st.sidebar.write("---")
st.sidebar.write("**All parameters can be tuned live to observe impact.**")

# ---------------- SIMULATION LOGIC ---------------- #
np.random.seed(42)

algo_sell_threshold = fair_price * (1 + algo_sell_threshold_pct / 100)
bid, ask = initial_bid, initial_ask
bid_history, ask_history, mid_history, pnl_history = [], [], [], []

human_position = 0
human_avg_price = None
human_order_posted = False
human_order_filled = False
trades = []

for t in range(steps):
    bid_history.append(bid)
    ask_history.append(ask)
    mid_history.append((bid + ask) / 2)

    # Human order posting
    if t == human_order_time:
        human_order_posted = True
        trades.append((t, 'Human', 'Post Limit Buy', human_limit_price))
    
    # Algo reaction when human order visible but not filled
    if human_order_posted and not human_order_filled:
        if human_limit_price >= ask:
            # Human immediately filled
            human_avg_price = ask
            human_position += 10
            trades.append((t, 'Human', 'Buy Filled', ask))
            human_order_filled = True
        else:
            # Algo reacts and starts buying aggressively (pushing price)
            bid += push_strength * 0.8
            ask = max(ask, bid + min_spread) + push_strength * 0.5
            
            trades.append((t, 'Algo', 'Buy Push', (bid + ask) / 2))
            
            # If price goes too high — sell back to human
            if (bid + ask) / 2 >= algo_sell_threshold:
                inflated_price = algo_sell_threshold
                trades.append((t, 'Algo', 'Sell to Human', inflated_price))
                human_avg_price = inflated_price
                human_position += 10
                human_order_filled = True
                bid, ask = initial_bid, initial_ask  # reset quotes
                trades.append((t, 'Algo', 'Reset Quotes', (bid, ask)))

    # Mean reversion to fair value
    bid += 0.05 * (fair_price - bid)
    ask += 0.05 * (fair_price - ask)

    # Noise to make it look realistic
    bid += np.random.normal(0, 0.3)
    ask += np.random.normal(0, 0.3)
    if ask - bid < min_spread:
        ask = bid + min_spread

    # Mark-to-market P&L
    if human_position > 0:
        pnl = (fair_price - human_avg_price) * human_position
    else:
        pnl = 0.0
    pnl_history.append(pnl)

# ---------------- DATA PREP ---------------- #
df = pd.DataFrame({
    "Time": range(steps),
    "Bid": bid_history,
    "Ask": ask_history,
    "Mid": mid_history,
    "Human_PnL": pnl_history
})

# ---------------- PLOT ---------------- #
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Time"], df["Bid"], label="Bid")
ax.plot(df["Time"], df["Ask"], label="Ask")
ax.plot(df["Time"], df["Mid"], label="Mid Price")
ax.axhline(fair_price, linestyle="--", color="gray", label=f"Fair Price = {fair_price}")
ax.axhline(algo_sell_threshold, linestyle=":", color="red", label=f"Algo Sell Threshold = {algo_sell_threshold:.1f}")

# mark trades
for (t, actor, action, price) in trades:
    if "Buy" in action:
        ax.scatter(t, price, color="blue", marker="x")
    elif "Sell" in action:
        ax.scatter(t, price, color="red", marker="D")

ax.set_xlabel("Time Step")
ax.set_ylabel("Price")
ax.set_title("Illiquid Option Market Dynamics")
ax.legend(loc="upper right")

# P&L plot
ax2 = ax.twinx()
ax2.plot(df["Time"], df["Human_PnL"], color="green", linestyle="--", label="Human Unrealized P&L")
ax2.set_ylabel("Human P&L")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left")

st.pyplot(fig)

# ---------------- DATA OUTPUT ---------------- #
st.subheader("📊 Trade Log")
st.dataframe(pd.DataFrame(trades, columns=["Time", "Actor", "Action", "Price"]))

st.subheader("📈 Price and P&L Data")
st.dataframe(df)

st.success(f"Final Human Position: {human_position} units at avg price {human_avg_price if human_avg_price else 'N/A'}")
st.info(f"Mark-to-Fair P&L: {pnl_history[-1]:.2f}")
