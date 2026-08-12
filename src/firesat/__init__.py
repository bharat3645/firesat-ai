"""FireSat-AI: hybrid CNN-LSTM wildfire risk forecasting for Alaska.

This package implements the core idea of the "Alaska Wildfire Prediction
Using Satellite Imagery" GSoC proposal: fuse multi-source satellite imagery
(Sentinel-1/2, Landsat, MODIS, VIIRS) with ERA5 reanalysis weather to derive
fire-risk indicators (NDVI, NBR, fuel-moisture proxies, thermal anomalies,
SAR backscatter) and feed them through a CNN + BiLSTM/GRU + attention model
that outputs probabilistic wildfire risk (No Risk / Moderate / High) at
1, 3, and 6 month horizons.
"""

__version__ = "0.1.0"
