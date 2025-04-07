# Startup Success Prediction with Dockerized Machine Learning

A full ML pipeline to predict startup success using scikit-learn, powered by Docker for reproducibility and portability. Includes model evaluation, saving/exporting results, and a clean DevOps-friendly structure.

---

## Problem Statement

Predict whether a startup is likely to succeed (IPO) based on features like funding amount, funding rounds, and category.

- Dataset: `startup_data.csv`
- Target: Binary (Success = IPO)

---

## Models Trained

This project trains and evaluates the following models using cross-validation:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost

---
---

---



## Results

Each model is evaluated using 5-fold cross-validation. Accuracy scores and standard deviation are printed and saved.

Sample Output:
Logistic Regression -> Mean Accuracy: 0.9738, Std Dev: 0.0003
Decision Tree -> Mean Accuracy: 0.9627, Std Dev: 0.0014
Random Forest -> Mean Accuracy: 0.9665, Std Dev: 0.0013
Gradient Boosting -> Mean Accuracy: 0.9734, Std Dev: 0.0005
AdaBoost -> Mean Accuracy: 0.9734, Std Dev: 0.0004

Exported evaluation results to 'evaluation_results.csv'
Best Model: Logistic Regression
Saved best model to 'best_model.pkl'


## Docker Setup

This project is containerized using Docker for easy deployment and reproducibility.

### Dockerfile Highlights

- **Base image**: Python 3.12 slim  
- **Dependencies**: Installed from `requirements.txt`  
- **Entrypoint**: Runs `main.py`

### Run with Docker
 Quick Start

```bash
docker build -t startup-success .
docker run --rm startup-success


How It Works
Preprocessing:

Numeric features are imputed and scaled

Categorical features are one-hot encoded

Model Training:

5 models are trained using scikit-learn pipelines

Each model is evaluated with cross-validation

Model Selection:

The best model (highest accuracy) is saved to best_model.pkl

All results exported to evaluation_results.csv

Project Structure
startup-success/
├── data/
│   └── startup_data.csv
├── main.py
├── Dockerfile
├── requirements.txt
├── evaluation_results.csv  # after running
├── best_model.pkl          # after running
└── README.md


```md
All Python dependencies are managed in `requirements.txt`. Key libraries include:

- `pandas`, `numpy`
- `scikit-learn`
- `joblib`

Jahnavi Sharma
Machine Learning & DevOps Enthusiast

