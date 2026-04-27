# 🩸 PlasmaCast: ML-Based Plasma Donor Demand Forecasting

> Predicting daily plasma donor demand across 10 major US cities using real-world weather, holiday, and behavioral data.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Accuracy](https://img.shields.io/badge/Accuracy-94.1%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is PlasmaCast?

PlasmaCast is an end-to-end machine learning system that forecasts daily plasma donor demand at plasma donation centers across 10 major US cities. It combines real historical weather data, US federal holiday calendars, and population-scaled donor baselines to train an XGBoost regression model capable of predicting donor counts with a **5.9% mean error rate**.

---

## Dashboard Preview

> Start the API and open `http://localhost:8000` to access the live dashboard.

The dashboard provides:
- 7-day donor demand forecast
- City-level insights 
- Model metrics & explainability
- Dark/light mode 

---

---

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.12                         |
| ML Model         | XGBoost Regressor                   |
| Feature Engineering | pandas, numpy                    |
| API Framework    | FastAPI + Uvicorn                   |
| Weather Data     | Open-Meteo API (free, no key)       |
| Holiday Calendar | Python `holidays` library           |
| Population Data  | US Census Bureau Vintage 2024       |
| Frontend         | HTML, CSS, JavaScript (vanilla)     |
| Model Storage    | joblib (.pkl)                       |

---

## Setup & Installation

### Prerequisites
- Python 3.12+
- pip
- Git
- Homebrew (Mac only, for XGBoost dependency)

### 1. Clone the repository
```bash
git clone https://github.com/PW13052003/plasmacast.git
cd plasmacast
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Mac users only** — XGBoost requires OpenMP:
```bash
brew install libomp
```

### 4. Generate the dataset
Fetches real historical weather from Open-Meteo for all 10 cities (2020–2023):
```bash
python src/data_gen.py
```

### 5. Engineer features
```bash
python src/features.py
```

### 6. Train the model
```bash
python src/model.py
```
This will print evaluation metrics and save the model to `src/data/model.pkl`.

### 7. Start the API and dashboard
```bash
uvicorn src.api:app --reload
```

Open your browser and go to: http://localhost:8000/

---

## Data Sources

| Source | Usage | Link |
|--------|-------|------|
| Open-Meteo Historical & Forecast API | Real daily weather (temp, precipitation) for all 10 cities | [open-meteo.com](https://open-meteo.com) |
| US Census Bureau Vintage 2024 | Population estimates used to scale donor baselines | [census.gov](https://census.gov) |
| Python `holidays` library | US federal holiday calendar for 2020–2023 | [pypi.org/project/holidays](https://pypi.org/project/holidays) |

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Regressor |
| Training Period | 2020 – 2022 (3 years) |
| Test Period | 2023 (1 year) |
| Train/Test Split | 75% / 25% |
| Training Rows | 10,820 |
| Test Rows | 3,650 |
| Features | 13 |
| MAE | 5.12 donors/day |
| RMSE | 7.95 donors/day |
| MAE as % of Mean | 5.9% |
| Overall Accuracy | 94.1% |



