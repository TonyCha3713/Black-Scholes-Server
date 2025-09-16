from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .bsm import price

@dataclass
class IVResult:
    sigma: float
    converged: bool
    iters: int

def implied_vol(target_price, S, K, T, r, option="call", q=0.0,
                lo=1e-6, hi=5.0, tol=1e-8, maxiter=100):
    """Solve BS price(sigma) = target_price via brentq; returns IVResult."""
    # Edge/degenerate cases
    intrinsic = (max(S-K,0.0) if option=="call" else max(K-S,0.0))
    if T <= 1e-12 or target_price <= intrinsic + 1e-14:
        return IVResult(0.0, True, 0)
    if target_price >= S:  # price can’t exceed S by no-arb
        return IVResult(hi, False, 0)

    def f(sig):
        return price(S,K,T,r,sig,option,q) - target_price

    try:
        sigma = brentq(f, lo, hi, xtol=tol, maxiter=maxiter)
        return IVResult(float(sigma), True, maxiter)
    except Exception:
        return IVResult(float("nan"), False, maxiter)
