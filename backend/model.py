import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

# Load dataset
df = pd.read_csv("data/covid.csv")


# Filter for one country (change if needed)
country_name = "India"
df = df[df["country"] == country_name]

# Sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Use daily new cases as signal
data = df[["daily_new_cases"]].fillna(0)

# Train Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(data)

#Save model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/anomaly_model.pkl", "wb"))

