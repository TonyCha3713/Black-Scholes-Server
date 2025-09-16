import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.bsm import price, greeks
from app.core.heatmap import grid_prices, grid_pnl
from app.db.db import SessionLocal, engine, Base
from app.db.models import Calculation, HeatmapPoint

Base.metadata.create_all(bind=engine)
st.set_page_config(page_title="Black–Scholes Pricer", layout="wide")

@st.cache_data(show_spinner=False)
def compute_grid(mode, S, K, T, r, sigma,
                 Smin, Smax, nS, sigmin, sigmax, nV,
                 option, purchase_price):
    if mode == "Value":
        S_vals, sig_vals, Z = grid_prices(S, K, T, r, sigma, Smin, Smax, nS, sigmin, sigmax, nV, option)
        title = f"{option.capitalize()} value"
    else:
        S_vals, sig_vals, Z = grid_pnl(S, K, T, r, sigma, Smin, Smax, nS, sigmin, sigmax, nV, option, purchase_price)
        title = f"{option.capitalize()} P&L (price - purchase)"
    return S_vals, sig_vals, Z, title

def main():
    st.title("Black–Scholes Option Pricer + Heatmaps")

    with st.sidebar:
        st.subheader("Inputs")
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Spot (S)", value=100.0, min_value=0.0)
            K = st.number_input("Strike (K)", value=100.0, min_value=0.0)
            r_pct = st.number_input("Risk-free r (%)", value=5.0, step=0.25)
        with col2:
            T_years = st.number_input("Time to expiry (years)", value=1.0, min_value=0.0, step=0.05, format="%.4f")
            sigma_pct = st.number_input("Volatility σ (%)", value=20.0, min_value=0.0, step=0.25)

        option = st.radio("Option side", ["call", "put"], horizontal=True)
        mode = st.radio("Heatmap mode", ["Value", "P&L"], horizontal=True)

        purchase = 0.0
        if mode == "P&L":
            purchase = st.number_input(f"Purchase price for {option}", value=10.0, min_value=0.0)

        st.subheader("Heatmap grid")
        S_span_pct = st.slider("Spot range around S (±%)", 1, 100, 30)
        sig_min_pct, sig_max_pct = st.slider("Vol range (%)", 1, 300, (5, 80))
        nS = st.slider("# Spot steps", 10, 150, 60)
        nV = st.slider("# Vol steps", 10, 150, 60)

        calc_btn = st.button("Calculate", type="primary")

    if calc_btn:
        r = r_pct/100.0; sigma = sigma_pct/100.0; T = T_years

        # Prices & Greeks
        call_val = price(S,K,T,r,sigma,"call")
        put_val  = price(S,K,T,r,sigma,"put")
        G = greeks(S,K,T,r,sigma)

        # Show summary
        left, right = st.columns([1,1])
        with left:
            st.metric("Call price", f"{call_val:,.4f}")
            st.metric("Put price",  f"{put_val:,.4f}")
        with right:
            st.write("**Greeks (at inputs)**")
            st.dataframe(pd.DataFrame(G).round(6))

        # Heatmap ranges
        Smin = S * (1 - S_span_pct/100.0)
        Smax = S * (1 + S_span_pct/100.0)
        sigmin = sig_min_pct/100.0
        sigmax = sig_max_pct/100.0

        S_vals, sig_vals, Z, ztitle = compute_grid(
            mode, S,K,T,r,sigma, Smin,Smax,nS, sigmin,sigmax,nV, option, purchase
        )

        # Save to DB
        with SessionLocal() as db:
            calc = Calculation(
                spot=S, strike=K, time_to_expiry=T, volatility=sigma, risk_free_rate=r,
                purchase_call=(purchase if (mode=="P&L" and option=="call") else None),
                purchase_put=(purchase if (mode=="P&L" and option=="put") else None),
            )
            db.add(calc); db.flush()
            # bulk insert (downsample if you like)
            rows = []
            for j, vv in enumerate(sig_vals):
                for i, ss in enumerate(S_vals):
                    if mode == "Value":
                        rows.append(HeatmapPoint(
                            calculation_id=calc.id, spot_shock=float(ss), vol=float(vv),
                            call_value=float(Z[j,i]) if option=="call" else None,
                            put_value=float(Z[j,i])  if option=="put"  else None
                        ))
                    else:
                        rows.append(HeatmapPoint(
                            calculation_id=calc.id, spot_shock=float(ss), vol=float(vv),
                            call_pnl=float(Z[j,i]) if option=="call" else None,
                            put_pnl=float(Z[j,i])  if option=="put"  else None
                        ))
            db.bulk_save_objects(rows)
            db.commit()

        # Plot
        zlabel = ztitle
        fig = px.imshow(
            Z, x=S_vals, y=sig_vals, origin="lower",
            labels=dict(x="Spot S", y="Vol σ", color=zlabel),
            color_continuous_scale=("RdYlGn" if mode=="P&L" else "Viridis"),
            zmin=(0.0 if mode=="Value" else None),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Past runs preview
        st.subheader("Recent runs")
        with SessionLocal() as db:
            q = db.query(Calculation).order_by(Calculation.id.desc()).limit(20).all()
            df = pd.DataFrame([{
                "id": c.id, "S": c.spot, "K": c.strike, "T": c.time_to_expiry,
                "sigma": c.volatility, "r": c.risk_free_rate,
                "p_call": c.purchase_call, "p_put": c.purchase_put, "created_at": c.created_at
            } for c in q])
            st.dataframe(df)
    else:
        st.info("Set inputs in the sidebar and click **Calculate**.")

if __name__ == "__main__":
    main()
