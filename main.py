import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("data/startup_data.csv")

# Clean funding column
df["funding_total_usd"] = (
    df["funding_total_usd"]
    .replace(r'[\$,]', '', regex=True)
    .replace('-', np.nan)
    .astype(float)
)

# Drop rows with missing values
df = df.dropna(subset=["funding_total_usd", "funding_rounds", "category_list", "status"])

# Binary target
df["success"] = df["status"].apply(lambda x: 1 if x.lower() == "ipo" else 0)

# Features and target
X = df[["funding_total_usd", "funding_rounds", "category_list"]]
y = df["success"]

# Preprocessing
numeric_features = ["funding_total_usd", "funding_rounds"]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_features = ["category_list"]
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Store results
results = []

for name, model in models.items():
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    scores = cross_val_score(clf, X, y, cv=5)
    results.append({
        "Model": name,
        "Accuracy": scores.mean(),
        "StdDev": scores.std()
    })
    print(f"{name} -> Mean Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
print("\n✅ Exported evaluation results to 'evaluation_results.csv'")

# Save best model
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")

# Re-train best model on full data before saving
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])
best_model.fit(X, y)
joblib.dump(best_model, "best_model.pkl")
print(f"✅ Saved best model to 'best_model.pkl'")

