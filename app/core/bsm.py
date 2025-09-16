import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm

EPS = 1e-12

def _d1_d2(S, K, T, r, sigma, q=0.0):
    # Broadcast-friendly; clamp to avoid division-by-zero
    T = np.maximum(T, EPS)
    sigma = np.maximum(sigma, EPS)
    d1 = (np.log(np.maximum(S, EPS)/np.maximum(K, EPS)) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def price(S, K, T, r, sigma, option="call", q=0.0):
    """European BS price (supports scalar or numpy arrays for S/sigma)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = float(r); sigma = np.asarray(sigma, dtype=float); q = float(q)

    # T->0: intrinsic value
    if np.all(T <= EPS):
        if option == "call":
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = exp(-r*T); disc_q = exp(-q*T)
    if option == "call":
        return disc_q*S*norm.cdf(d1) - disc_r*K*norm.cdf(d2)
    else:
        return disc_r*K*norm.cdf(-d2) - disc_q*S*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0):
    """Return dict of Delta/Gamma/Vega/Theta/Rho for call & put."""
    S = float(S); K = float(K); T = max(float(T), EPS); r = float(r); sigma = max(float(sigma), EPS); q = float(q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r*T); disc_q = np.exp(-q*T)
    pdf = norm.pdf(d1)

    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega  = S * disc_q * pdf * np.sqrt(T)

    delta_c = disc_q * norm.cdf(d1)
    delta_p = delta_c - disc_q

    theta_c = (-S*disc_q*pdf*sigma/(2*np.sqrt(T))) - r*K*disc_r*norm.cdf(d2) + q*S*disc_q*norm.cdf(d1)
    theta_p = (-S*disc_q*pdf*sigma/(2*np.sqrt(T))) + r*K*disc_r*norm.cdf(-d2) - q*S*disc_q*norm.cdf(-d1)

    rho_c =  K*T*disc_r*norm.cdf(d2)
    rho_p = -K*T*disc_r*norm.cdf(-d2)

    return {
        "call": {"delta": delta_c, "gamma": gamma, "vega": vega, "theta": theta_c, "rho": rho_c},
        "put":  {"delta": delta_p, "gamma": gamma, "vega": vega, "theta": theta_p, "rho": rho_p},
    }
