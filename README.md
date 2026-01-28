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

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Kaggle API:

   - Create account at [kaggle.com](https://www.kaggle.com)
   - Go to Settings → API → Generate New Token
   - Move the downloaded file:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download the dataset:

```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
unzip data/raw/store-sales-time-series-forecasting.zip -d data/raw/
rm data/raw/store-sales-time-series-forecasting.zip
```

## Usage

1. Run `notebooks/01_exploration.ipynb` to explore the data
2. Run `notebooks/02_training.ipynb` to train the model

## License

MIT
