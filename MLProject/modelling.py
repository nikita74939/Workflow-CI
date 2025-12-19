import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import json

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("Telco-Customer-Churn_preprocessing.csv")

    X = data.drop("Churn", axis=1).astype("float64")
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    # Parameter model
    C = 1.0
    solver = "liblinear"
    class_weight = "balanced"

    # Mulai training dan logging
    with mlflow.start_run():

        # Aktifkan autolog dari sklearn
        mlflow.autolog()

        # Buat dan latih model
        model = LogisticRegression(
            C=C,
            solver=solver,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)

        # Hitung metrik utama
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Buat dan log confusion matrix sebagai artifact gambar
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix - Logistic Regression")
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")

        # Buat dan log koefisien model ke csv (optional tapi kamu bisa simpan sebagai artifact)
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": model.coef_[0]
        }).sort_values(by="coefficient", key=abs, ascending=False)
        coef_df.to_csv("lr_coefficients.csv", index=False)
        mlflow.log_artifact("lr_coefficients.csv")

        # --- Minimal dua artifact selain autolog() ---

        # 1) Estimator HTML
        estimator_html = f"""
        <html>
        <head><title>Estimator Summary</title></head>
        <body>
        <h2>Logistic Regression Coefficients</h2>
        {coef_df.to_html(index=False)}
        </body>
        </html>
        """
        with open("estimator.html", "w") as f:
            f.write(estimator_html)
        mlflow.log_artifact("estimator.html")

        # 2) Metric info JSON
        metric_info = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
        with open("metric_info.json", "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        # Buat input example dan signature
        input_example = X_train.head().astype("float64")
        signature = infer_signature(
            model_input=input_example,
            model_output=model.predict(input_example)
        )

        # Log model dengan manual agar nama folder model sesuai
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )