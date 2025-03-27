import pandas as pd

# Load the dataset
df = pd.read_csv("ecg.csv")

# Get a sample row (without the target column)
sample_input = df.iloc[0, :-1].tolist()

# Print in JSON format
import json
print(json.dumps({"features": sample_input}))

