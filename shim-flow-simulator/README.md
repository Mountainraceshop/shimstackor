# Shim Calculator Engineering — v2 MVP

A runnable MVP for a physics-led suspension damper simulator.

## What this build includes

- FastAPI backend + browser UI
- Platform-aware modelling for:
  - Forks: coil, KYB PSF1, KYB PSF2, Showa air, WP air
  - Shocks: monotube and twin-tube approximations (e.g. KYB/Showa/WP and Ohlins TTX/K-Tech style)
- Valve-stage separation:
  - piston stack
  - base valve stack
  - mid-valve stack
- Low-speed and rebound clicker restriction modelling
- High-speed poppet/blow-off approximation
- Port-count and port-diameter based flow capacity
- Pressure / force / shim-lift / flow-split curves over shaft velocity
- Reservoir pressure probe and cavitation margin estimate
- Spring/chamber progression curve for fork and shock modes
- Dyno CSV parser endpoint and chart overlay in UI

## What this build does **not** yet include

- Full CFD / FSI
- Multi-body vehicle dynamics
- Full transient (time-domain) valve inertia and hysteresis solver
- Detailed check-plate contact mechanics
- Real shim stress/fatigue maps
- Full temperature transient model

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

This is a solid scaffold for the engineering product brief and is intended to be extended. The solver is intentionally transparent and heavily commented so it can be tuned against dyno data.

## API endpoints

- `POST /api/simulate`  
  Runs the physics model and returns summary, compression/rebound velocity curves, and spring/chamber progression.

- `POST /api/parse-dyno-csv`  
  Accepts `{ "csv_text": "..." }` and parses velocity/force columns for dyno overlay.

## Render deployment (important)

The app is inside `shim-flow-simulator/`, not the repo root.

If configuring manually in Render:

- **Root Directory**: `shim-flow-simulator`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`

This repository also includes a `render.yaml` at repo root with the same values.
