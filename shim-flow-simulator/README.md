# Shim Calculator Engineering — MVP

A runnable MVP for a physics-led suspension damper simulator.

## What this build includes

- FastAPI backend
- Browser UI served by FastAPI
- Geometry, port area, simple bleed/adjuster modelling
- Simplified shim-stack opening model
- Force/pressure/lift/flow-split outputs over shaft velocity
- Fork air spring / oil height progression estimate
- Professional engineering-style single-page layout

## What this build does **not** yet include

- Full CFD / FSI
- Multi-body vehicle dynamics
- Full dyno CSV import/export
- Full high-speed poppet dynamics
- Real shim stress maps
- Temperature transient solver

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
