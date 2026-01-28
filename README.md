# Sales Forecast

Time series forecasting project using AutoGluon to predict store sales.

## Overview

This project uses the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset from Kaggle to predict future sales using AutoGluon's TimeSeriesPredictor.

## Dataset

The dataset contains sales data from Corporación Favorita, a large Ecuadorian grocery retailer, including:

- Historical sales data
- Store metadata
- Product family information
- Oil prices (Ecuador is oil-dependent)
- Holiday events

## Project Structure

```
sales-forecast/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
├── models/             # Saved models
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## License

MIT
