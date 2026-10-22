# Customer Churn Intelligence System using ML

_Designed and developed by Prathmesh Girme_

End-to-end ML app: train a churn model, serve it through a Streamlit dashboard, deploy it online.

## Features
- Single-customer prediction with probability gauge, risk level and retention tips
- Adjustable decision threshold
- Batch CSV scoring with validation, charts, and downloadable results
- Feature importance and what-if analysis
- Cached model loading, input validation, logging, Docker support

## Tech stack
Python, scikit-learn, pandas, Streamlit, Plotly, Docker

## Run locally
```bash
pip install -r requirements.txt
python train_model.py          # skip if you already have customer_churn_model.pkl
streamlit run app.py
```

## Docker
```bash
docker build -t churn-dashboard .
docker run -p 8501:8501 churn-dashboard
```

## Deploy (Streamlit Community Cloud - free)
1. Push this folder (including the `.pkl` file) to a public GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, click **New app**.
3. Select the repo, branch `main`, main file `app.py`, then **Deploy**.

## Input format for batch prediction
CSV with columns: `Age, Tenure, MonthlyCharges, SupportCalls`

## Notes
Pin the same scikit-learn version used to create the `.pkl` file in `requirements.txt`.
