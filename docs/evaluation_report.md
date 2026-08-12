# FireSat-AI — Evaluation Report

**Data**: synthetic (see [Data honesty](#data-honesty) below) · **Regions**: Interior
Alaska (Fairbanks), Kenai Peninsula · **Period**: 2015-01 to 2024-12 (120 months/region)
· **Split**: chronological, trailing 20% per region held out as validation (no shuffling
— validation is strictly later in time than training, per region) · **Model**:
`FireSatNet` (ResNet-style CNN + channel attention → BiLSTM → temporal attention → 3
horizon heads) · **Training**: 10 epochs, Adam (lr=1e-3, weight_decay=1e-4), batch size
16, inverse-frequency class-weighted cross-entropy, seed 42.

Regenerate this report's numbers with:
```bash
python scripts/generate_demo_data.py     # writes data/processed/ (seed 5, calibrated)
python scripts/train_demo.py             # writes models/checkpoints/firesat_demo.pt
                                          # and docs/eval_metrics.json
```

## Headline numbers

| Horizon | Val n | Accuracy | Macro F1 | Macro Precision | Macro Recall | Fire Recall | False-Alarm Rate |
|---|---|---|---|---|---|---|---|
| 1 month  | 38 | 0.842 | 0.305 | 0.281 | 0.333 | 0.00 | 0.00 |
| 3 months | 38 | 0.553 | 0.237 | 0.184 | 0.333 | 0.00 | 0.00 |
| 6 months | 38 | 0.316 | 0.174 | 0.118 | 0.333 | 0.154 | 0.00 |

*Fire recall* = fraction of validation months with a realized fire (Moderate or High
ground truth) that the model flagged as elevated risk (Moderate/High predicted).
*False-alarm rate* = fraction of "High"-predicted months where no fire actually
occurred. See `docs/eval_metrics.json` for full confusion matrices and per-epoch loss
curves.

## Majority-class baseline comparison

The honest comparison for any rare-event classifier is "how much better than always
predicting the most common class":

| Horizon | Model accuracy | Always-predict-majority-class accuracy | Model beats baseline? |
|---|---|---|---|
| 1 month  | 0.842 | 0.842 (majority = No Risk, 32/38) | **No — exactly tied** |
| 3 months | 0.553 | 0.553 (majority = No Risk, 21/38) | **No — exactly tied** |
| 6 months | 0.316 | 0.368 (majority = *High*, 14/38)  | **No — underperforms** |

## What this actually shows

Reading the confusion matrices directly (`docs/eval_metrics.json`): the model predicts
**"No Risk" for essentially every validation month at every horizon**, with a handful of
"Moderate" predictions appearing only at the 6-month horizon. It never predicts "High"
at all. In other words, on this run, **FireSatNet has not learned to discriminate risk
classes from the input features — it has collapsed to (approximately) the training
majority class.**

This is a real, reproducible result, not a bug being papered over. Reporting it plainly
is the point of an "honest evaluation" rather than an overclaimed demo. The likely
causes, roughly in order of expected impact:

1. **Very few positive examples.** With a calibrated-but-still-modest synthetic ignition
   rate (~8–14 fire events per region over 10 years), the *1-month* horizon label is
   "a fire starts in exactly the next calendar month" — a genuinely rare event even
   before considering it's a 3-way classification problem. Real-world short-horizon
   wildfire ignition prediction has the same structural difficulty.
2. **Small dataset relative to model capacity.** 146 training windows (after
   chronological splitting and the 24-month lookback requirement) is small for a
   CNN+BiLSTM+attention model with ~140K parameters; inverse-frequency class weighting
   alone was not sufficient to prevent majority-class collapse at this scale.
3. **Simple class-weighted cross-entropy is a weak tool for this level of imbalance.**
   Techniques like focal loss, minority oversampling/SMOTE-for-sequences, or
   two-stage (detect-then-classify) formulations are the standard next step and are not
   yet implemented here.
4. **Ten epochs is short.** Validation loss increases after epoch 1 (see
   `docs/eval_metrics.json::history`), consistent with the model fitting the majority
   class quickly and then mostly overfitting minor noise rather than learning the rare
   classes — more epochs alone would likely not fix this without also addressing (1)–(3).

## Concrete next steps (not yet implemented)

- Focal loss or class-balanced loss (Cui et al. 2019) in place of plain weighted CE.
- Oversample or upweight fire-adjacent months more aggressively during batching.
- Expand the synthetic ignition record (more years and/or more regions) so each horizon
  has enough positive examples for a meaningful train/val split.
- Once live data is wired in (see README → "Using real satellite/weather data"), real
  NDVI/NBR/SAR/ERA5 signal may carry more learnable structure than the synthetic
  generator's simplified dryness process — this evaluation should be re-run against real
  acquisitions before drawing any conclusions about real-world Alaska fire risk.
- Backtest predicted risk directly against NASA FIRMS detection density and AICC
  perimeter acreage (not just the internal synthetic ignition label) once live data is
  wired in, per the original proposal's evaluation plan.

## Data honesty

The dataset behind these numbers is **synthetic**, produced by
`firesat.data.synthetic.SyntheticDataGenerator`: real Alaska monthly climate normals
drive seasonal NDVI/temperature/humidity/precipitation cycles, and a logistic
dryness-driven process stochastically ignites "fires" that in turn drive NBR drops,
thermal anomalies, and synthetic FIRMS-like detections. It is designed to give the
pipeline genuine (if modest and noisy) precursor signal to learn from, not to mimic
real Alaska fire statistics precisely. **Nothing in this report should be read as a
claim about real-world Alaska wildfire risk.** It documents how the *pipeline* performs
on data it was designed to be exercised against — see README → "Scope & honesty notes"
and `src/firesat/data/synthetic.py` for the full generation methodology, and swap in
live Earth Engine / ERA5 / FIRMS / AICC data (clients already implemented in
`src/firesat/data/`) before drawing any real-world conclusions.

## Other simplifications worth restating here

- `fire_weather_danger_proxy` / `fuel_moisture_index` (`src/firesat/features/weather.py`)
  are transparent monotonic heuristics, not the Canadian FWI System or US NFDRS.
- `gradient_input_saliency` (`src/firesat/models/interpret.py`) is single-pass
  gradient×input, not Integrated Gradients/SHAP.
- Historical perimeter records without a precise ignition date are attributed to July
  (`src/firesat/data/perimeters.py::to_monthly_burned_area`).
