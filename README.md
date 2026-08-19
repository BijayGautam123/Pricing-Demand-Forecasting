# Pricing Demand Forecasting

Pricing Demand Forecasting is a portfolio proof of concept for exploring pricing scenarios with synthetic data. A linear regression model estimates case demand from price, discount, holiday, weekday, and recent-demand inputs. The Streamlit app then calculates simulated revenue and profit and uses Gemini to explain the scenario in business terms.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pricing-demand-forecasting-g5s2phfhngkgwcd5anvq6m.streamlit.app/)

**Live App:** [Open Pricing Demand Forecasting](https://pricing-demand-forecasting-g5s2phfhngkgwcd5anvq6m.streamlit.app/)

## Project Structure

```text
.
|-- app.py
|-- data/
|   `-- pricing_portfolio_data.csv
|-- models/
|   |-- pricing_model_package.pkl
|   `-- cases_model.pkl
|-- notebooks/
|   `-- pricing_model_analysis.ipynb
|-- .streamlit/
|   `-- secrets.toml.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

The Streamlit app uses `models/pricing_model_package.pkl`. The additional `cases_model.pkl` artifact is retained for reference but is not loaded by the current app.

## Setup

Create and activate a Python 3.11 virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local Streamlit secrets file from the included template:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Replace the placeholder in `.streamlit/secrets.toml` with a Gemini API key. The real secrets file is ignored by Git and must not be committed.

## Run the Application

From the repository root:

```powershell
python -m streamlit run app.py
```

## Model Development

The analysis notebook is located at `notebooks/pricing_model_analysis.ipynb`. It loads the synthetic dataset, engineers lag and rolling-demand features, and performs a chronological train/test split. Linear Regression, Random Forest, and XGBoost are compared using R-squared, MSE, and RMSE. The notebook also reviews Random Forest and XGBoost feature importance, retains standard and time-series cross-validation for Linear Regression, and checks multicollinearity with VIF.

Saved notebook results include:

- Linear Regression test R-squared: 0.907; RMSE: 126.4 cases
- Random Forest test R-squared: 0.884; RMSE: 141.3 cases
- XGBoost test R-squared: 0.863; RMSE: 153.2 cases
- Linear Regression time-series cross-validation R-squared: 0.913

Linear Regression is retained for the Streamlit application because it produced the strongest testing performance and a smaller training-to-testing performance gap than the tree-based alternatives. The included dataset contains 517 synthetic daily pricing observations with price, discount, holiday, cases-sold, and weekday fields.

## Current Scope

This project is a portfolio demonstration, not a production pricing system. Its forecasts depend on the included dataset and a static recent-demand baseline stored with the model. Predictions should be interpreted as scenario estimates rather than guaranteed business outcomes.
