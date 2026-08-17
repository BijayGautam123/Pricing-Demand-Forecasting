# ML Pricing Core

ML Pricing Core is a Streamlit application for exploring pricing scenarios. A linear regression model estimates case demand from price, discount, holiday, weekday, and recent-demand inputs. The app then calculates revenue and profit and uses Gemini to explain the scenario in business terms.

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

The analysis notebook is located at `notebooks/pricing_model_analysis.ipynb`. It loads the repository dataset, engineers lag and rolling-demand features, performs a chronological train/test split, evaluates linear regression, and packages the model used by the app.

Saved notebook results include:

- Test R-squared: 0.907
- Test RMSE: 126.4 cases
- Time-series cross-validation R-squared: 0.913

The included dataset contains 517 daily pricing observations with price, discount, holiday, cases-sold, and weekday fields.

## Current Scope

This project is a portfolio demonstration, not a production pricing system. Its forecasts depend on the included dataset and a static recent-demand baseline stored with the model. Predictions should be interpreted as scenario estimates rather than guaranteed business outcomes.
