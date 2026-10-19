import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import joblib
import os

# 1. Load Data
df = pd.read_csv('data/heart_cleaned.csv')
X = df.drop('target', axis=1)
y = df['target']

# Binary target: 0 = no disease, 1-4 = disease. Let's binarize it for standard classification.
y = y.apply(lambda x: 1 if x > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocessing Pipeline
preprocessor = StandardScaler()

# 3. Define Models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 4. MLflow Tracking
mlflow.set_experiment("Heart_Disease_MLOps_Exp")

best_auc = 0.0
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Create pipeline
        pipe = Pipeline([
            ('scaler', preprocessor),
            ('classifier', model)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Log all model parameters to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_params({
            key: str(value)
            for key, value in model.get_params().items()
        })
            
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        
        # Log ROC curve
        os.makedirs("artifacts", exist_ok=True)
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f"{name} - ROC Curve")
        fig.tight_layout()
        roc_path = f"artifacts/{name}_roc_curve.png"
        fig.savefig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path="plots")
        plt.close(fig)

        # Log confusion matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        cm_path = f"artifacts/{name}_confusion_matrix.png"
        fig.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        plt.close(fig)

        # Log Model and Preprocessor
        mlflow.sklearn.log_model(pipe, "model")
        
        print(f"{name} - AUC: {auc:.4f}")
        
        # Track best model
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipe

# 5. Save Best Model Locally for Docker (Reproducibility)
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipeline, 'models/best_model.pkl')
print(f"Best model ({best_model_name}) saved to models/best_model.pkl")