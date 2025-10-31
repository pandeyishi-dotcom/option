import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="Illiquid Option Market Simulator (Live)", layout="wide")
st.title("📈 Illiquid Option Market — Live Simulation (No Matplotlib/Plotly)")

st.sidebar.header("Simulation parameters")
fair_price = st.sidebar.number_input("Fair price of option", value=40.0, min_value=0.01, step=1.0)
initial_bid = st.sidebar.number_input("Initial Algo Bid", value=20.0, min_value=0.0, step=1.0)
initial_ask = st.sidebar.number_input("Initial Algo Ask", value=80.0, min_value=initial_bid+0.1, step=1.0)
time_steps = st.sidebar.slider("Time steps", min_value=10, max_value=300, value=60, step=10)
human_order_time = st.sidebar.slider("Human arrival (time step)", min_value=0, max_value=100, value=5)
human_order_size = st.sidebar.number_input("Human order size (contracts)", value=1.0, min_value=0.01, step=0.1)
volatility = st.sidebar.number_input("Volatility (noise)", value=2.0, min_value=0.0, step=0.1)
mean_reversion_speed = st.sidebar.number_input("Mean reversion speed", value=0.08, min_value=0.0, max_value=1.0, step=0.01)
quote_adjust_speed = st.sidebar.number_input("Quote adjust speed", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
animation_speed = st.sidebar.slider("Animation speed (seconds per step)", 0.01, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("This demo is purely educational — not trading advice.")

run_sim = st.sidebar.button("Run live simulation")

if run_sim:
    np.random.seed(42)

    bid, ask = initial_bid, initial_ask
    mid_price = (bid + ask) / 2
    human_position, human_cost = 0.0, 0.0
    human_avg_price = None

    df = pd.DataFrame(columns=["time", "mid_price", "bid", "ask", "pnl"])
    chart_placeholder = st.empty()
    pnl_placeholder = st.empty()
    event_log = st.container()

    with event_log:
        st.subheader("🧾 Event Log")
        event_text = st.empty()

    events = []

    for t in range(time_steps):
        # Simulate normal mid-price drift
        shock = np.random.normal(scale=volatility)
        mid_price += mean_reversion_speed * (fair_price - mid_price) + shock * 0.1
        bid += quote_adjust_speed * (mid_price - bid) * 0.1
        ask += quote_adjust_speed * (mid_price - ask) * 0.1

        if ask - bid < 0.2:
            ask = bid + 0.2

        # Human enters the market
        if t == human_order_time:
            fill_price = ask
            human_position += human_order_size
            human_cost += fill_price * human_order_size
            human_avg_price = fill_price
            events.append(f"t={t}: Human bought {human_order_size} @ {fill_price:.2f}")
            bid += 0.5  # algo reaction
            ask += 0.5
            mid_price = (bid + ask) / 2

        # Occasional external trade impact
        if np.random.rand() < 0.03:
            bid -= 0.2
            ask -= 0.1
            events.append(f"t={t}: External seller hit the bid @ {bid:.2f}")
            mid_price = (bid + ask) / 2

        # Compute human P&L (mark-to-market)
        pnl = 0
        if human_position > 0:
            pnl = human_position * (bid - human_avg_price)

        new_row = {"time": t, "mid_price": mid_price, "bid": bid, "ask": ask, "pnl": pnl}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Plot live chart
        chart_placeholder.line_chart(df.set_index("time")[["mid_price", "bid", "ask"]])

        # Show PnL info
        pnl_placeholder.metric("💰 Human Unrealized P&L", f"{pnl:.2f}")

        # Show last few events
        if len(events) > 5:
            show_events = events[-5:]
        else:
            show_events = events
        event_text.write("\n".join(show_events))

        time.sleep(animation_speed)

    # Final liquidation
    liquidation_price = bid
    realized_pnl = human_position * liquidation_price - human_cost
    st.success(f"Final human liquidation at bid {liquidation_price:.2f}, realized P&L = {realized_pnl:.2f}")

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="live_sim_quotes.csv",
        mime="text/csv"
    )
else:
    st.info("Adjust the parameters on the left, then click **Run live simulation** to see the market evolve.")
