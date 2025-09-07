from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('online_gaming_behavior_dataset.csv')

# Add derived feature
df["TotalWeeklyPlaytime"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"] / 60

# 1. Label Encoding for categorical variables
categorical_vars = ['Gender', 'Location', 'GameGenre',
                    'InGamePurchases', 'GameDifficulty', 'EngagementLevel']

# --------------------------
# 1. Prepare Data
# --------------------------
X_selected = df[["TotalWeeklyPlaytime", "SessionsPerWeek",
                 "AvgSessionDurationMinutes", "PlayerLevel","Age"]]
y = df['EngagementLevel']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder (optional: if you need to decode predictions later)
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# --------------------------
# 2. Split Data
# --------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.3, random_state=42
)

# --------------------------
# 3. Define Model
# --------------------------
best_xgb = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=90,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=6,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    eval_metric="mlogloss"
)

# --------------------------
# 4. Train Model
# --------------------------
best_xgb.fit(x_train, y_train)

# --------------------------
# 5. Evaluate Model
# --------------------------
start_time = time.time()

train_pred = best_xgb.predict(x_train)
test_pred = best_xgb.predict(x_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

end_time = time.time()
test_time = end_time - start_time

print(f"Test Prediction Time: {test_time:.4f} seconds")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# --------------------------
# 6. Save Trained Model
# --------------------------
best_xgb.save_model("xgb_model.json")
print("✅ Model saved as xgb_model.json")
