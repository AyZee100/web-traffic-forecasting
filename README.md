# Web Traffic Time Series: Forecasting & Anomaly Detection

Time Series Analysis project — forecasting and anomaly detection on the [Kaggle Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset (daily views of ~145,000 Wikipedia pages, 2015-07-01 → 2017-09-10).

We aggregate all pages into one **total daily traffic** series and run the full workflow: exploration → decomposition → SARIMA/Prophet forecasting → anomaly detection (4 methods) → re-forecasting after cleaning.

## Key results

| Model | RMSE before cleaning | RMSE after cleaning |
|---|---|---|
| SARIMA(1,1,2)(0,1,1)₇ | **12.65M** | 16.06M |
| Prophet | 38.11M | 22.16M (−42%) |

- Series is non-stationary (ADF p ≈ 0.08); first difference is stationary. Strong weekly seasonality (Monday peak, Friday trough).
- 20 consensus anomalies (flagged by ≥2 of 4 methods) map to real events: **US election** (302M views), the **Rio Olympics** period (all-time peak, 317M views), **David Bowie's death**, and the Oscars mix-up.
- Cleaning anomalies dramatically helps Prophet (global curve fit) but slightly hurts SARIMA — differencing is inherently robust, so cleaning was unnecessary for it.
- Note: the LSTM autoencoder is not bit-reproducible across environments, so its flag count (and thus the consensus set) can shift by a day or two between runs; the other three methods are deterministic.

## Repository structure

```
├── web_traffic_forecasting.ipynb   # full analysis notebook (executed, with outputs)
├── data/daily_total.csv            # aggregated daily total views (cached)
├── figs/                           # exported figures
├── slides.pptx                     # presentation
├── requirements.txt
└── README.md
```

## Reproducing

1. (Optional — only to re-aggregate from raw) Download `train_2.csv` from the [Kaggle competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data) and place it at `train_2.csv/train_2.csv`. Otherwise the notebook uses the cached `data/daily_total.csv`.
2. Install dependencies and run:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace web_traffic_forecasting.ipynb
```

Runtime ≈ 3–5 min (SARIMA grid search + Prophet + LSTM autoencoder training).

## Methods

**Exploration:** missing-value analysis, ADF stationarity test, ACF/PACF.
**Decomposition:** classical additive and robust STL (period 7); trend/seasonal strength per Hyndman & Athanasopoulos.
**Forecasting:** SARIMA (AIC grid search) and Prophet, 60-day holdout, RMSE/MAE/AIC/BIC.
**Anomaly detection:** robust Z-score on STL residuals, SARIMA standardized residuals, Isolation Forest, LSTM autoencoder — combined by consensus voting (≥2 methods).
**Re-forecasting:** consensus anomalies imputed with same-weekday ±4-week median; both models refit and re-evaluated against the original test data.

## Team

Aymane Zeroual - Otmane ZAIM - Omar Tanji
