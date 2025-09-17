# Black-Scholes Options Pricer — P&L-Only (Streamlit + NumPy)

A clean, interview-ready options project focused on **trader intuition**: two tabs (**Call** / **Put**) showing prices, compact Greeks, and crisp **grid-style P&L heatmaps** with a break-even contour. Inputs are simple and fast; UI is tight and production-minded.

> **Live demo**: _add your Streamlit/Render link_  
> **Repo topics**: `quant` `options` `black-scholes` `streamlit` `numpy` `plotly` `sqlalchemy`

---

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Implementation Breakdown](#implementation-breakdown)
- [Configuration](#configuration)
- [Numerical Notes](#numerical-notes)
- [Performance Notes](#performance-notes)
- [Testing (Suggested)](#testing-suggested)
- [Troubleshooting](#troubleshooting)
- [Why P&L-Only?](#why-pl-only)
- [Roadmap (Optional)](#roadmap-optional)
- [Resume Snippets](#resume-snippets)
- [License](#license)

---

## Features

- **P&L-only** heatmaps (no raw “value” mode) — green = profit, red = loss
- Two tabs: **Call** and **Put**  
  Each tab shows: price, purchase price, base P&L, **compact Greeks**, heatmap, and 1D P&L slices
- **Grid-style** heatmaps (visible cell boundaries, no smoothing) with **break-even contour** and **ATM line**
- Slices use the **base** S and σ (no extra sliders) for a tidy, quick UX
- **Vectorized** computation (NumPy) + **Streamlit caching** for snappy updates
- **Optional** silent persistence to a relational DB (SQLite by default) for later analysis

---

## Screenshots

Add these after you run the app and capture images:

```
/docs/screenshot-call.png
/docs/screenshot-put.png
```

![Call Tab](docs/screenshot-call.png)  
![Put Tab](docs/screenshot-put.png)

---

## Project Structure

```
black-scholes/
├─ app/
│  ├─ ui.py                      # Streamlit app (P&L-only, Call/Put tabs)
│  ├─ core/
│  │  ├─ bsm.py                  # Black-Scholes price + Greeks
│  │  └─ heatmap.py              # Vectorized P&L grids (grid_pnl)
│  └─ db/                        # (optional) persistence
│     ├─ db.py                   # SQLAlchemy engine/session
│     └─ models.py               # Calculation, HeatmapPoint
├─ requirements.txt
├─ README.md
└─ bsm.db                        # (optional) SQLite file created at runtime
```

> If you don’t want persistence, omit `app/db/*` and keep `PERSIST=0` (default).

---

## Requirements

- Python **3.10+** (3.11 works great)
- pip / venv
- Internet connection (to load Plotly/Streamlit assets in the browser)

---

## Quick Start

```bash
# 1) Create & activate a virtualenv
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# (If you don't have a requirements.txt yet)
# pip install streamlit numpy scipy pandas plotly sqlalchemy

# 3) Run FROM THE PROJECT ROOT (folder containing app/)
streamlit run app/ui.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Usage

**Sidebar**

- Spot `S`, Strike `K`, Time to expiry `T` (years), Risk-free rate `r` (%), Volatility `σ` (%)
- Purchase price (**Call**) and Purchase price (**Put**) for P&L
- Heatmap grid limits (conservative defaults to keep it responsive):
  - Spot range around S (±%)
  - Volatility range (%)
  - Grid density (# steps for S and σ)

**Tabs**

- **Call**: Price, purchase price, **base P&L**, compact **Greeks** (Δ, Γ, Vega, Θ, ρ), then:
  - **P&L Heatmap** (S × σ): base point marker, ATM line, **break-even** contour
  - **Slices** (no sliders): P&L vs **S** at base σ, and P&L vs **σ** at base S
  - **Download** CSV of the P&L grid
- **Put**: Same layout for the put side

---

## Implementation Breakdown

### Math Core (`app/core/bsm.py`)
- Black-Scholes closed form (European call/put) with guards:
  - Clamp tiny `T`/`σ` to avoid division by zero; at `T → 0` return **intrinsic value**
  - Guard `log(S/K)` against zeros
- Greeks returned as a dict: Delta, Gamma, Vega (per 1.0 vol), Theta (per year), Rho

### P&L Grids (`app/core/heatmap.py`)
- `grid_pnl(...)`: build 1D `S_vals` and `sig_vals` → `np.meshgrid` → compute price minus purchase price over the grid
- Returns 2D arrays (rows = σ, columns = S) to match Plotly’s expectations

### UI (`app/ui.py`)
- **P&L-only** (no “value” toggle)
- **Two tabs** (Call / Put)
- **Compact Greeks** table so it doesn’t dominate the layout
- **Grid-style** Plotly heatmaps:
  - `zsmooth=false`, `xgap=1`, `ygap=1` for visible cell boundaries
  - **Break-even** contour (Z=0), **ATM** line, base (S, σ) marker
  - Heatmap height increased (≈720 px by default) for readability
- **Slices** fixed at base σ and base S for cleanliness
- **Caching** of grid calculations via `@st.cache_data`

### Optional Persistence (`app/db`)
- Enable with `PERSIST=1`:
  - `Calculation` (inputs + purchase prices)
  - `HeatmapPoint` (downsampled grid points for storage efficiency)
- Default DB is **SQLite** (`sqlite:///./bsm.db`), easily switchable via env var

---

## Configuration

**Enable persistence**

```bash
export PERSIST=1
# Optional: switch DB (Postgres example)
export DB_URL=postgresql+psycopg2://user:pass@host:5432/dbname
```

Ensure `app/db/db.py` reads `DB_URL` or adjust accordingly.

**Heatmap size**

The heatmap is set larger in code (`height≈720`). Increase if desired inside `make_grid_heatmap(...)`.

---

## Numerical Notes

- **Units**
  - `T` in years
  - `r`, `σ` entered in % (UI) → converted to decimals internally
  - **Theta** is per year; **Vega** is per 1.0 vol (not per 1%)
- **Stability**
  - Tiny `T`/`σ` clamps, intrinsic at `T → 0`, guarded `log(S/K)`
- **P&L color semantics**
  - Diverging palette centered at 0 (green profit / red loss)
  - Break-even line explicitly shown

---

## Performance Notes

- Grid math is **vectorized** (NumPy) — no Python loops
- **Streamlit cache** avoids recomputation when inputs repeat
- Conservative default grid sizes keep the UI responsive
- If persisting, grids are **downsampled** before insert to keep the DB small

---

## Testing (Suggested)

Create `tests/` with:

- Known price fixtures (ATM, deep ITM/OTM)
- Properties:
  - Call price increases with **S** and **σ**
  - Put–call parity (within tolerance)
  - Gamma > 0, Vega ≥ 0

Run:

```bash
pytest -q
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'app'`**  
  Run Streamlit from the **project root** (the folder containing `app/`):
  ```bash
  streamlit run app/ui.py
  ```

- **Charts are slow**  
  Reduce grid sizes (# steps) or narrow S/σ ranges in the sidebar.

- **DB errors**  
  Keep `PERSIST=0` (default) or ensure `app/db/*` exists and `DB_URL` is valid.

---

## Why P&L-Only?

- Matches how traders think (“Where do I make/lose money?”).
- Cleaner UI and stronger interview narrative.
- You can keep a “value” branch for pedagogy/debugging if you like.

---

## Roadmap (Optional)

- Greeks heatmaps (Delta, Gamma, Vega)
- Auto-fill purchase prices to current BS prices (so base P&L starts at 0)
- API split with FastAPI (`/price`, `/heatmap`) + Docker
- Option chain import + simple IV surface (K–T)

---

## Resume Snippets

- Built a Streamlit **P&L-only** Black-Scholes pricer with vectorized NumPy kernels and **grid-style heatmaps** (break-even contours, ATM line).  
- Implemented robust numerics (intrinsic at **T→0**, σ clamps, parity/property tests) and a compact UI (Call/Put tabs, tidy Greeks).  
- (Optional) Persisted calculation inputs and downsampled grids to a relational DB via SQLAlchemy.

---

## License

MIT — see `LICENSE`.
