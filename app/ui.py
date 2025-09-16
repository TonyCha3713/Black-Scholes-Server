import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

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

def make_grid_heatmap(S_vals, sig_vals, Z, mode, ztitle, show_text=False):
    """
    Grid-style heatmap with visible cell boundaries and sensible ticks.
    - show_text=True overlays numbers in each cell (auto-hides if grid is too big).
    """
    colorscale = "Viridis" if mode == "Value" else "RdYlGn"
    zmid = 0.0 if mode == "P&L" else None
    zmin = 0.0 if mode == "Value" else None

    # Build the heatmap
    hm = go.Heatmap(
        z=Z,
        x=S_vals,
        y=sig_vals,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zsmooth=False,      # no interpolation — crisp cells
        xgap=1,             # gaps draw visible gridlines between cells
        ygap=1,
        colorbar=dict(title=ztitle),
        hovertemplate="S=%{x:.4g}<br>σ=%{y:.4g}<br>"+ztitle+"=%{z:.4g}<extra></extra>",
    )

    fig = go.Figure(data=hm)

    # Axes styling
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Spot S",
        yaxis_title="Vol σ",
        margin=dict(l=50, r=50, t=30, b=40),
    )

    # Thin the number of ticks so labels don’t overlap
    def thin(vals, max_ticks=12):
        step = max(1, len(vals) // max_ticks)
        return vals[::step]

    fig.update_xaxes(showgrid=True, gridcolor="LightGray",
                     tickmode="array", tickvals=thin(S_vals),
                     tickformat="~g")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray",
                     tickmode="array", tickvals=thin(sig_vals),
                     tickformat="~g")

    # Optional per-cell text overlay (only if not too dense)
    if show_text and (len(S_vals) * len(sig_vals) <= 2000):
        text = [[f"{Z[j, i]:.2f}" for i in range(len(S_vals))] for j in range(len(sig_vals))]
        fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=10))

    return fig

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
        fig = make_grid_heatmap(S_vals, sig_vals, Z, mode, ztitle=zlabel, show_text=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Set inputs in the sidebar and click **Calculate**.")

if __name__ == "__main__":
    main()
