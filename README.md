# FireSat-AI

**Hybrid CNN-LSTM + attention wildfire risk forecasting for Alaska**, fusing multi-source
satellite imagery (Sentinel-1/2, Landsat, MODIS, VIIRS) with ERA5 reanalysis weather to
produce probabilistic wildfire risk classifications (**No Risk / Moderate / High**) at
**1, 3, and 6 month horizons**.

Built as a GSoC-style MVP for the "Alaska Wildfire Prediction Using Satellite Imagery"
proposal. Deliberately scoped to two well-studied regions rather than all of Alaska, so
the pipeline, model, and evaluation stay tractable, reproducible, and honestly reported
rather than overclaimed.

> **Status**: working end-to-end MVP. Ships with a physically-motivated **synthetic**
> data generator so the full pipeline (data → features → model → API → dashboard) runs
> with zero credentials. Real acquisition clients for Earth Engine / ERA5 / NASA FIRMS /
> Alaska Fire Service are implemented and documented but not wired to live network calls
> in this checkout — see [Scope & honesty notes](#scope--honesty-notes).

## What's in the box

| Layer | What it does | Where |
|---|---|---|
| **Data acquisition** | Real clients for Sentinel-1/2/Landsat/MODIS (Earth Engine), ERA5-Land (CDS/GEE), NASA FIRMS, Alaska Fire Service perimeters — plus a synthetic generator for offline dev | `src/firesat/data/` |
| **Feature engineering** | NDVI, NBR, ΔNBR, NDMI, EVI, SAR radar vegetation index / cross-pol ratio / soil-moisture proxy, fuel-moisture & fire-weather-danger indices | `src/firesat/features/` |
| **Model** | ResNet-style CNN (+ squeeze-excite channel attention) → BiLSTM/GRU → additive temporal attention → 3 multi-horizon classification heads | `src/firesat/models/` |
| **Training** | Chronological train/val split, class-weighted cross-entropy, checkpointing | `src/firesat/training/` |
| **Evaluation** | Per-horizon accuracy/F1/precision/recall, confusion matrices, wildfire-specific fire-recall & false-alarm-rate backtest | `src/firesat/training/evaluate.py` |
| **Interpretability** | Channel-attention & temporal-attention summaries, gradient×input saliency, plotting helpers | `src/firesat/models/interpret.py` |
| **API** | FastAPI backend: regions, current risk, historical risk trend, fire history | `src/firesat/api/` |
| **Dashboard** | Leaflet GIS map + risk panels + attention visualizations, vanilla JS, no build step | `dashboard/` |
| **Ops** | Dockerfile, docker-compose, GitHub Actions CI (lint + tests + pipeline smoke test) | `Dockerfile`, `.github/workflows/ci.yml` |

## Study regions

Rather than "all of Alaska," this MVP scopes to two fire-active, well-documented regions:

1. **Interior Alaska (Fairbanks / Yukon-Tanana Uplands)** — boreal black-spruce forest,
   one of the most fire-active landscapes in North America.
2. **Kenai Peninsula** — spruce-beetle-killed forest with elevated fuel load and
   wildland-urban-interface risk.

See `src/firesat/config.py::REGIONS` for exact bounding boxes.

## Architecture

```
                 monthly (C=6, H=16, W=16) feature stack
                              │
              ┌───────────────────────────────┐
              │  ResNet-style CNN encoder      │
              │  + squeeze-excite channel      │──▶ per-channel attention (interpretable)
              │    attention                   │
              └───────────────────────────────┘
                              │  spatial embedding (per month)
                              ▼
              concat with ERA5 weather features (temp, RH, wind, precip)
                              │
              ┌───────────────────────────────┐
              │  Bidirectional LSTM / GRU      │   24-month lookback sequence
              └───────────────────────────────┘
                              │
              ┌───────────────────────────────┐
              │  Additive temporal attention   │──▶ per-month attention (interpretable)
              └───────────────────────────────┘
                              │  context vector
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
            1-month head  3-month head  6-month head
            (No Risk / Moderate / High, softmax)
```

Spatial channels (in order): `ndvi, nbr, lst_anomaly, sar_vv, sar_vh, fuel_moisture_proxy`.
Weather channels: `temp_2m_c, relative_humidity_pct, wind_speed_ms, precipitation_mm`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 1. Generate the offline synthetic dataset (or point at real data, see below)
python scripts/generate_demo_data.py

# 2. Train FireSatNet (a few minutes on CPU)
python scripts/train_demo.py

# 3. Serve the API + dashboard
uvicorn firesat.api.main:app --reload
# → dashboard at http://localhost:8000/
# → API docs at http://localhost:8000/docs
```

A pre-generated demo dataset (`data/processed/`) and trained checkpoint
(`models/checkpoints/firesat_demo.pt`) are committed to this repo, so
`uvicorn firesat.api.main:app` works immediately without steps 1–2 — they're
there if you want to regenerate with a different seed/date range or retrain.

### Docker

```bash
docker compose up --build
# → http://localhost:8000/
```

### Tests

```bash
pytest -q                 # 49 tests: indices, SAR, weather, model, dataset, training, API
ruff check src tests scripts
```

## Using real satellite/weather data

The synthetic generator (`src/firesat/data/synthetic.py`) is a physically-motivated
stand-in — seasonal NDVI green-up/senescence, ERA5-like climate normals for each region,
and a stochastic ignition process driven by a "dryness score" so the model has genuine
precursor signal to learn from — but it is **not real data**. To go live:

1. `pip install -e ".[geo]"` (adds `earthengine-api`, `cdsapi`)
2. `earthengine authenticate` (free Google account)
3. Request a free [NASA FIRMS MAP_KEY](https://firms.modaps.eosdis.gov/api/area/) and set
   `FIRMS_MAP_KEY` (see `.env.example`)
4. Download historical perimeters from the
   [Alaska Fire Service / AICC open data portal](https://fire.ak.blm.gov) as GeoJSON
5. Wire `firesat.data.pipeline._run_live_pipeline` to call
   `EarthEngineClient` / `ERA5Client` / `FIRMSClient` / `PerimeterLoader` (all implemented
   in `src/firesat/data/`, each documented with the exact collections/endpoints used) and
   assemble their outputs into the same `SyntheticRegionDataset`-shaped contract —
   nothing downstream (features, model, training, API, dashboard) needs to change.

## API reference

| Endpoint | Description |
|---|---|
| `GET /api/health` | Backend readiness + loaded regions |
| `GET /api/regions` | Study region metadata + GeoJSON geometry |
| `GET /api/risk/{region_id}` | Latest risk prediction (all 3 horizons) + attention |
| `GET /api/risk/{region_id}/history?months=24` | Historical predicted-risk trend |
| `GET /api/risk/{region_id}/fire-history` | Backtest ground truth: recorded ignitions |

Full interactive docs at `/docs` (FastAPI/Swagger) once the server is running.

## Repository layout

```
firesat-ai/
├── src/firesat/
│   ├── config.py            # regions, feature schema, constants
│   ├── data/                # acquisition: GEE, ERA5, FIRMS, perimeters, synthetic, pipeline
│   ├── features/             # NDVI/NBR/SAR/weather feature engineering + normalization
│   ├── models/                # CNN encoder, temporal encoder, attention, FireSatNet, interpret
│   ├── training/               # dataset windowing, train loop, evaluation/backtest
│   ├── inference.py            # loads checkpoint + data, serves predictions
│   └── api/                    # FastAPI app + routers + schemas
├── dashboard/                  # Leaflet GIS dashboard (vanilla JS, static)
├── scripts/                    # generate_demo_data.py, train_demo.py
├── tests/                      # 49 tests across every layer
├── docs/evaluation_report.md   # honest metrics + limitations
├── data/processed/             # committed small synthetic demo dataset
├── models/checkpoints/         # committed trained demo checkpoint
├── Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml    # lint + test + pipeline smoke test
```

## Scope & honesty notes

This is an MVP built to demonstrate a rigorous applied-climate-tech pipeline, not a
production fire-forecasting system. Specifically:

- **Synthetic-by-default data.** The shipped dataset is generated, not observed. It is
  physically motivated (real Alaska climate normals, seasonal vegetation cycles, a
  dryness-driven stochastic ignition process) so the pipeline and model are genuinely
  exercised end-to-end, but risk numbers from the demo checkpoint describe the synthetic
  world, not real Alaska fire risk. See `docs/evaluation_report.md`.
- **The shipped demo checkpoint does not clearly beat a majority-class baseline** on
  this small synthetic dataset (it ties it at 1 and 3 months, underperforms it at 6
  months) — reported plainly, with a concrete diagnosis and next steps, in
  `docs/evaluation_report.md` rather than hidden or dressed up. The pipeline (data →
  features → model → training → evaluation → API → dashboard) is genuinely complete and
  working end-to-end; what's honestly *not* yet demonstrated is that the model
  outperforms a naive baseline on this data volume.
- **Two regions, not all of Alaska.** Chosen for fire activity and data availability, not
  because the approach doesn't generalize — extending `REGIONS` in `config.py` is the
  only change needed to add more.
- **Weather-danger indices are simplified.** `fire_weather_danger_proxy` /
  `fuel_moisture_index` are transparent, monotonic heuristics, *not* the Canadian Fire
  Weather Index System or US NFDRS, which need daily (not monthly) inputs and carried
  multi-day state. Documented explicitly in `src/firesat/features/weather.py`.
- **Saliency, not full attribution.** `gradient_input_saliency` is a single-backward-pass
  gradient×input approximation, not Integrated Gradients/SHAP.
- **Fire-month attribution for perimeters is approximate.** Historical perimeter records
  without a precise ignition date are attributed to July (the historical peak of the
  Alaska fire season) — flagged in `perimeters.py` and the evaluation report.

## License

MIT — see [LICENSE](LICENSE).
