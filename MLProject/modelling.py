import mlflow
import mlflow.sklearn
import pandas as pd
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    # 1. Tangkap Parameter dari MLProject
    C_param = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    solver_param = sys.argv[2] if len(sys.argv) > 2 else 'liblinear'
    
    print(f"Starting training with C={C_param}, solver={solver_param}")

    # 2. Load Data
    csv_path = "Telco-Customer-Churn_preprocessing.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        sys.exit(1)
        
    data = pd.read_csv(csv_path)
    X = data.drop("Churn", axis=1).astype("float64")
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    # 3. Training & Logging
    with mlflow.start_run():
        mlflow.autolog()

        model = LogisticRegression(
            C=C_param,
            solver=solver_param,
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Training Complete. Accuracy: {acc}")
        
        mlflow.log_metric("accuracy_manual", acc)