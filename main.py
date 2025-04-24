import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Dict, Optional

def predict_startup_success(
    startup_name: str,
    funding_total_usd: float,
    funding_rounds: int,
    category: str,
    model_path: str = "best_model.pkl"
) -> Dict:
    """Predict success probability for a single startup."""
    # Load the trained model
    model = joblib.load(model_path)

    # Create a DataFrame with single startup data
    startup_data = pd.DataFrame({
        "funding_total_usd": [funding_total_usd],
        "funding_rounds": [funding_rounds],
        "category_list": [category]
    })

    # Get prediction probability
    prob = model.predict_proba(startup_data)[0][1]

    return {
        "startup_name": startup_name,
        "success_probability": round(prob * 100, 2),
        "prediction": "Likely to IPO" if prob >= 0.5 else "Unlikely to IPO"
    }

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

# Initialize variables that will be used in train_and_evaluate_models
results_df = None
best_model = None

def train_and_evaluate_models(visualize: bool = False, save_viz_path: Optional[str] = None):
    """Train and evaluate models, with optional visualization."""
    global df, X, y, models, results_df, best_model

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

    # Visualize results if requested
    if visualize:
        try:
            from visualize import visualize_model_comparison
            # Create visualizations directory if it doesn't exist
            if save_viz_path and not os.path.exists(os.path.dirname(save_viz_path)):
                os.makedirs(os.path.dirname(save_viz_path))
            visualize_model_comparison(save_path=save_viz_path)
        except ImportError:
            print("Warning: Visualization module not found. Skipping visualization.")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {str(e)}")

if __name__ == "__main__":
    # Train and evaluate models with visualization
    train_and_evaluate_models(visualize=True, save_viz_path="visualizations/model_comparison.png")

    # Example prediction
    prediction = predict_startup_success(
        startup_name="TechStartup Inc",
        funding_total_usd=1000000,
        funding_rounds=2,
        category="Software"
    )

    print("\n🔮 Startup Prediction:")
    print(f"Startup: {prediction['startup_name']}")
    print(f"Success Probability: {prediction['success_probability']}%")
    print(f"Prediction: {prediction['prediction']}")

    # Visualize prediction
    try:
        from visualize import visualize_prediction
        visualize_prediction(prediction, save_path="visualizations/example_prediction.png")
    except ImportError:
        print("Warning: Visualization module not found. Skipping visualization.")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {str(e)}")
