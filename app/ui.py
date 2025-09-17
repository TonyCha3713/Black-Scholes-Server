# app/ui.py

import math
import os
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.bsm import price, greeks
from app.core.heatmap import grid_pnl  # P&L-only grids

# --- Optional persistence (silent; no "recent runs" UI) -----------------------
PERSIST = os.getenv("PERSIST", "0") in ("1", "true", "True")

SessionLocal = None
Calculation = None
HeatmapPoint = None
Base = None
engine = None

if PERSIST:
    try:
        from app.db.db import SessionLocal as _SessionLocal, Base as _Base, engine as _engine
        from app.db.models import Calculation as _Calculation, HeatmapPoint as _HeatmapPoint

        SessionLocal = _SessionLocal
        Calculation = _Calculation
        HeatmapPoint = _HeatmapPoint
        Base = _Base
        engine = _engine
        Base.metadata.create_all(bind=engine)
    except Exception:
        PERSIST = False


# --- Streamlit page config ----------------------------------------------------
st.set_page_config(page_title="Black–Scholes Pricer — P&L", layout="wide")


# --- Cache: compute P&L grids for both sides ---------------------------------
@st.cache_data(show_spinner=False)
def compute_pnl_grids(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    Smin: float,
    Smax: float,
    nS: int,
    sigmin: float,
    sigmax: float,
    nV: int,
    purchase_call: float,
    purchase_put: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    P&L-only: compute call and put P&L grids.
    Returns: (S_vals, sig_vals, Z_call_pnl, Z_put_pnl, title_call, title_put)
    """
    S_vals, sig_vals, Z_call = grid_pnl(
        S, K, T, r, sigma, Smin, Smax, int(nS), sigmin, sigmax, int(nV),
        option="call", purchase_price=purchase_call
    )
    _, _, Z_put = grid_pnl(
        S, K, T, r, sigma, Smin, Smax, int(nS), sigmin, sigmax, int(nV),
        option="put", purchase_price=purchase_put
    )
    title_call = "Call P&L (price - purchase_call)"
    title_put  = "Put P&L (price - purchase_put)"
    return S_vals, sig_vals, Z_call, Z_put, title_call, title_put


# --- Plot helpers -------------------------------------------------------------
def make_grid_heatmap(
    S_vals: np.ndarray,
    sig_vals: np.ndarray,
    Z: np.ndarray,
    *,
    ztitle: str,
    S_base: float,
    sigma_base: float,
    K: float,
    height: int = 720,  
) -> go.Figure:
    """
    Grid-style P&L heatmap:
      - Diverging RdYlGn (green = profit, red = loss)
      - Break-even (Z=0) contour line
      - Base point and ATM marker
    """
    hm = go.Heatmap(
        z=Z,
        x=S_vals,
        y=sig_vals,
        colorscale="RdYlGn",
        zmid=0.0,           # center diverging colormap at 0 P&L
        zsmooth=False,      # crisp cells
        xgap=1,
        ygap=1,
        colorbar=dict(title=ztitle),
        hovertemplate="S=%{x:.6g}<br>σ=%{y:.6g}<br>" + ztitle + "=%{z:.6g}<extra></extra>",
    )

    fig = go.Figure(data=hm)
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Spot S",
        yaxis_title="Vol σ",
        margin=dict(l=50, r=50, t=30, b=40),
        height=height,
    )

    def thin(vals: np.ndarray, max_ticks: int = 12) -> np.ndarray:
        step = max(1, len(vals) // max_ticks)
        return vals[::step]

    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)",
                     tickmode="array", tickvals=thin(S_vals), tickformat="~g")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)",
                     tickmode="array", tickvals=thin(sig_vals), tickformat="~g")

    # Base point and ATM line
    fig.add_trace(go.Scatter(
        x=[S_base], y=[sigma_base], mode="markers",
        marker=dict(size=10, symbol="x-thin", line=dict(width=2)),
        name="Base", hovertemplate="Base<br>S=%{x:.6g}<br>σ=%{y:.6g}<extra></extra>",
    ))
    fig.add_vline(x=K, line_dash="dot", line_color="gray",
                  annotation_text="ATM", annotation_position="top", opacity=0.6)

    # Break-even contour (Z == 0)
    fig.add_trace(go.Contour(
        z=Z, x=S_vals, y=sig_vals, showscale=False,
        contours=dict(start=0.0, end=0.0, size=1.0, coloring="lines"),
        line=dict(color="black", width=2), name="Break-even", hoverinfo="skip",
    ))

    return fig


def slice_plot(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="lines", name=y_label))
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=30, t=40, b=40),
        title=title,
    )
    return fig


def compact_greeks_table(side_greeks: dict) -> pd.DataFrame:
    """Return a small, tidy 1-row DataFrame for compact display."""
    # Order + formatting-friendly names
    order = [("Delta", "delta"), ("Gamma", "gamma"), ("Vega", "vega"), ("Theta", "theta"), ("Rho", "rho")]
    data = {label: side_greeks[key] for (label, key) in order if key in side_greeks}
    return pd.DataFrame([data]).round(6)


# --- Main UI ------------------------------------------------------------------
def main():
    st.title("Black–Scholes Option Pricer — P&L")

    # ----- Sidebar inputs (rows)
    with st.sidebar:
        st.subheader("Inputs")
        S = st.number_input("Spot (S)", value=100.0, min_value=0.0, step=1.0)
        K = st.number_input("Strike (K)", value=100.0, min_value=0.0, step=1.0)
        T_years = st.number_input("Time to expiry (years)", value=1.0, min_value=0.0, step=0.05, format="%.6f")
        r_pct = st.number_input("Risk-free r (%)", value=5.0, step=0.25, format="%.6f")
        sigma_pct = st.number_input("Volatility σ (%)", value=20.0, min_value=0.0, step=0.25, format="%.6f")

        st.subheader("Purchase prices (for P&L)")
        purchase_call = st.number_input("Purchase price (Call)", value=10.0, min_value=0.0, step=0.1)
        purchase_put  = st.number_input("Purchase price (Put)",  value=10.0, min_value=0.0, step=0.1)

        st.subheader("Heatmap grid (limited)")
        # Tighter ranges to keep UI responsive and avoid huge inputs
        S_span_pct = st.slider("Spot range around S (±%)", 5, 50, 20)
        sig_min_pct, sig_max_pct = st.slider("Vol range (%)", 5, 150, (10, 60))
        nS = st.slider("# Spot steps", 20, 80, 40)
        nV = st.slider("# Vol steps", 20, 80, 40)

        calc_btn = st.button("Calculate", type="primary")

    # Tabs: Call and Put
    tab_call, tab_put = st.tabs(["Call", "Put"])

    if not calc_btn:
        st.info("Set inputs in the sidebar and click **Calculate**.")
        return

    # ----- Normalize / validate
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0
    T = T_years

    # Build ranges
    Smin = max(0.0, S * (1 - S_span_pct / 100.0))
    Smax = S * (1 + S_span_pct / 100.0)
    sigmin = min(sigma, sig_min_pct / 100.0)
    sigmax = max(sigma, sig_max_pct / 100.0)
    if sigmin == sigmax:
        sigmax = sigmin + 0.0001

    # ----- Scalar prices, base P&L, and Greeks
    call_price = float(price(S, K, T, r, sigma, "call"))
    put_price  = float(price(S, K, T, r, sigma, "put"))
    call_pnl_base = call_price - purchase_call
    put_pnl_base  = put_price  - purchase_put
    G = greeks(S, K, T, r, sigma)  # dict with 'call' and 'put'

    call_greeks_df = compact_greeks_table(G["call"])
    put_greeks_df  = compact_greeks_table(G["put"])

    # ----- Grids (P&L for both sides)
    S_vals, sig_vals, Z_call, Z_put, title_call, title_put = compute_pnl_grids(
        S, K, T, r, sigma, Smin, Smax, nS, sigmin, sigmax, nV, purchase_call, purchase_put
    )

    # ----- Optional persistence (no UI surface) — store P&L only
    if PERSIST and SessionLocal is not None:
        try:
            with SessionLocal() as db:
                calc = Calculation(
                    spot=float(S), strike=float(K), time_to_expiry=float(T),
                    volatility=float(sigma), risk_free_rate=float(r),
                    purchase_call=float(purchase_call), purchase_put=float(purchase_put),
                )
                db.add(calc)
                db.flush()

                target_points = 2000  # cap rows written
                stride = max(1, int(math.sqrt((len(S_vals) * len(sig_vals)) / max(1, target_points))))

                rows = []
                for j in range(0, len(sig_vals), stride):
                    vv = float(sig_vals[j])
                    for i in range(0, len(S_vals), stride):
                        ss = float(S_vals[i])
                        rows.append(
                            HeatmapPoint(
                                calculation_id=calc.id, spot_shock=ss, vol=vv,
                                call_pnl=float(Z_call[j, i]), put_pnl=float(Z_put[j, i]),
                            )
                        )
                if rows:
                    db.bulk_save_objects(rows)
                db.commit()
        except Exception:
            pass  # silent if DB unavailable

    # ----- CALL TAB -----
    with tab_call:
        st.subheader("Overview (Call)")
        left, right = st.columns([2, 1])
        with left:
            m1, m2, m3 = st.columns(3)
            m1.metric("Call price", f"{call_price:,.4f}")
            m2.metric("Purchase price (Call)", f"{purchase_call:,.4f}")
            m3.metric("Base P&L (Call)", f"{call_pnl_base:,.4f}")
        with right:
            st.caption("Greeks at input (Θ per year; Vega per 1.0 vol)")
            st.table(call_greeks_df)  # compact, non-interactive

        st.subheader("Heatmap (Call P&L)")
        fig_call = make_grid_heatmap(
            S_vals, sig_vals, Z_call, ztitle=title_call, S_base=S, sigma_base=sigma, K=K, height=720
        )
        st.plotly_chart(fig_call, use_container_width=True)

        st.subheader("Slices (Call P&L)")
        # No sliders: use base sigma and base S
        j_call = int(np.argmin(np.abs(sig_vals - sigma)))
        i_call = int(np.argmin(np.abs(S_vals - S)))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                slice_plot(
                    x_vals=S_vals, y_vals=Z_call[j_call, :],
                    x_label="Spot S", y_label="P&L",
                    title=f"vs S (σ={sig_vals[j_call]:.4f})",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                slice_plot(
                    x_vals=sig_vals, y_vals=Z_call[:, i_call],
                    x_label="Vol σ", y_label="P&L",
                    title=f"vs σ (S={S_vals[i_call]:.4f})",
                ),
                use_container_width=True,
            )

        # CSV download for Call P&L
        df_call = pd.DataFrame(Z_call, index=pd.Index(sig_vals, name="sigma"), columns=pd.Index(S_vals, name="S"))
        st.download_button(
            label="Download Call P&L grid (CSV)",
            data=df_call.to_csv().encode("utf-8"),
            file_name=f"grid_call_pnl_{int(S)}_{int(K)}.csv",
            mime="text/csv",
        )

    # ----- PUT TAB -----
    with tab_put:
        st.subheader("Overview (Put)")
        left, right = st.columns([2, 1])
        with left:
            m1, m2, m3 = st.columns(3)
            m1.metric("Put price", f"{put_price:,.4f}")
            m2.metric("Purchase price (Put)", f"{purchase_put:,.4f}")
            m3.metric("Base P&L (Put)", f"{put_pnl_base:,.4f}")
        with right:
            st.caption("Greeks at input (Θ per year; Vega per 1.0 vol)")
            st.table(put_greeks_df)

        st.subheader("Heatmap (Put P&L)")
        fig_put = make_grid_heatmap(
            S_vals, sig_vals, Z_put, ztitle=title_put, S_base=S, sigma_base=sigma, K=K, height=720
        )
        st.plotly_chart(fig_put, use_container_width=True)

        st.subheader("Slices (Put P&L)")
        # No sliders: use base sigma and base S
        j_put = int(np.argmin(np.abs(sig_vals - sigma)))
        i_put = int(np.argmin(np.abs(S_vals - S)))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                slice_plot(
                    x_vals=S_vals, y_vals=Z_put[j_put, :],
                    x_label="Spot S", y_label="P&L",
                    title=f"vs S (σ={sig_vals[j_put]:.4f})",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                slice_plot(
                    x_vals=sig_vals, y_vals=Z_put[:, i_put],
                    x_label="Vol σ", y_label="P&L",
                    title=f"vs σ (S={S_vals[i_put]:.4f})",
                ),
                use_container_width=True,
            )

        # CSV download for Put P&L
        df_put = pd.DataFrame(Z_put, index=pd.Index(sig_vals, name="sigma"), columns=pd.Index(S_vals, name="S"))
        st.download_button(
            label="Download Put P&L grid (CSV)",
            data=df_put.to_csv().encode("utf-8"),
            file_name=f"grid_put_pnl_{int(S)}_{int(K)}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
