import numpy as np
from .bsm import price

def grid_prices(S_base, K, T, r, sigma_base,
                S_min, S_max, nS,
                sig_min, sig_max, nV,
                option="call", q=0.0):
    S_vals   = np.linspace(S_min,   S_max,   int(nS))
    sig_vals = np.linspace(sig_min, sig_max, int(nV))
    SS, VV = np.meshgrid(S_vals, sig_vals, indexing="xy")  # X=S, Y=sigma

    # Vectorized price; K,T,r,q are scalars
    P = price(SS, K, T, r, VV, option=option, q=q)
    return S_vals, sig_vals, P

def grid_pnl(S_base, K, T, r, sigma_base,
             S_min, S_max, nS,
             sig_min, sig_max, nV,
             option="call", purchase_price=0.0, q=0.0):
    S_vals, sig_vals, P = grid_prices(S_base,K,T,r,sigma_base,
                                      S_min,S_max,nS,sig_min,sig_max,nV,option,q)
    pnl = P - purchase_price
    return S_vals, sig_vals, pnl
