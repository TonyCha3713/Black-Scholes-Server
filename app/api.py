from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, List
import numpy as np
from .core.bsm import price
from .core.iv import implied_vol
from .core.heatmap import grid_prices, grid_pnl

app = FastAPI(title="BSM API")

class PriceReq(BaseModel):
    S: float; K: float; T: float; r: float; sigma: float
    option: Literal["call","put"]="call"
    q: float = 0.0

class PriceResp(BaseModel):
    value: float

@app.post("/price", response_model=PriceResp)
def price_endpoint(req: PriceReq):
    v = float(price(req.S, req.K, req.T, req.r, req.sigma, req.option, req.q))
    return {"value": v}

class IVReq(BaseModel):
    target_price: float; S: float; K: float; T: float; r: float
    option: Literal["call","put"]="call"; q: float=0.0

class IVResp(BaseModel):
    sigma: float; converged: bool

@app.post("/implied-vol", response_model=IVResp)
def iv_endpoint(req: IVReq):
    res = implied_vol(req.target_price, req.S, req.K, req.T, req.r, req.option, req.q)
    return {"sigma": res.sigma, "converged": res.converged}

class GridReq(BaseModel):
    S: float; K: float; T: float; r: float; sigma: float
    Smin: float; Smax: float; nS: int
    sigmin: float; sigmax: float; nV: int
    option: Literal["call","put"]="call"
    mode: Literal["Value","PnL"]="Value"
    purchase: float = 0.0

class GridResp(BaseModel):
    S: List[float]; sigmas: List[float]; Z: List[List[float]]  # 2D list

@app.post("/heatmap", response_model=GridResp)
def heatmap_endpoint(req: GridReq):
    if req.mode == "Value":
        S_vals, sig_vals, Z = grid_prices(req.S, req.K, req.T, req.r, req.sigma,
                                          req.Smin, req.Smax, req.nS, req.sigmin, req.sigmax, req.nV, req.option)
    else:
        S_vals, sig_vals, Z = grid_pnl(req.S, req.K, req.T, req.r, req.sigma,
                                       req.Smin, req.Smax, req.nS, req.sigmin, req.sigmax, req.nV,
                                       req.option, req.purchase)
    return {"S": S_vals.tolist(), "sigmas": sig_vals.tolist(), "Z": Z.tolist()}
