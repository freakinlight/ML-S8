import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("ecg.csv")

# Features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
model_filename = "model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(rf_model, file)

print(f"✅ Model training complete! Saved as {model_filename}")

