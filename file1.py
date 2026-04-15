# ===============================
# CAR INSURANCE CLAIM PREDICTION
# FINAL OPTIMIZED MODEL
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("train.csv")
print("Dataset shape:", df.shape)

# -------------------------
# DROP ID
# -------------------------
if "policy_id" in df.columns:
    df.drop(columns=["policy_id"], inplace=True)

# -------------------------
# FIX STRING → NUMERIC 🔥
# -------------------------
def extract_number(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

for col in ["max_power", "gross_weight"]:
    if col in df.columns:
        df[col] = df[col].apply(extract_number)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------
# FEATURE ENGINEERING 🚀
# -------------------------
if "max_power" in df.columns and "gross_weight" in df.columns:
    df["engine_per_weight"] = df["max_power"] / (df["gross_weight"] + 1)

if "age_of_car" in df.columns and "age_of_policyholder" in df.columns:
    df["age_ratio"] = df["age_of_car"] / (df["age_of_policyholder"] + 1)

# -------------------------
# TARGET
# -------------------------
TARGET = "is_claim"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nClass distribution:")
print(y.value_counts(normalize=True))

# -------------------------
# COLUMN TYPES
# -------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("\nNumeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# -------------------------
# PREPROCESSING
# -------------------------
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# -------------------------
# SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# HANDLE IMBALANCE
# -------------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# -------------------------
# MODEL 🚀 (OPTIMIZED XGBOOST)
# -------------------------
pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])

# -------------------------
# TRAIN
# -------------------------
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

print("\n===== RESULTS =====")
print(classification_report(y_val, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_val, y_proba), 4))

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(pipeline, "car_claim_numeric_model.pkl")
print("\nModel saved as car_claim_numeric_model.pkl")