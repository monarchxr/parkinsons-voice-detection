import numpy as np
import pickle
import pandas as pd

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

order = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'HNR', 'NHR']



def predict(features: dict):

    try:
        x = pd.DataFrame([features]).reindex(columns=order, fill_value=0)

        x_scaled = scaler.transform(x)

        prob = model.predict_proba(x_scaled)[0][1]
        pred = model.predict(x_scaled)[0]

        return{
            "prediction": int(pred),
            "probability": float(prob)
        }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")