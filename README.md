# SmartCart — Streamlit Customer Segmentation

SmartCart is a customer segmentation and RFM analysis application built from the SmartCart notebook.

## Run locally

Place these files in the same folder:

- `app.py`
- `smartcart_customers.csv`
- `smartcart_model_artifacts.pkl`
- `requirements.txt`

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model artifact

Before running Streamlit for the first time, execute the final deployment cell in the notebook. It saves the encoder, scaler, prediction classifier, cluster profile, recommendations, and RFM thresholds to `smartcart_model_artifacts.pkl`.

## Deploy

Push the project files to a GitHub repository and create a Streamlit Community Cloud app using `app.py` as the entry point.
