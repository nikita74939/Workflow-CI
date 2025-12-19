# Workflow-CI  
## Implementasi Continuous Integration (CI) untuk Machine Learning dengan MLflow dan Docker

Repository ini dibuat untuk memenuhi **Kriteria 3: Membuat Workflow CI**, yang bertujuan mengimplementasikan proses *continuous integration* pada proyek machine learning menggunakan **MLflow**, **GitHub Actions**, dan **Docker Hub**.

---

## 📊 Artefak MLflow

Artefak yang dihasilkan dan disimpan antara lain:
- Confusion Matrix (`training_confusion_matrix.png`)
- Koefisien Logistic Regression (`lr_coefficients.csv`)
- Ringkasan metrik model (`metric_info.json`)
- Ringkasan estimator (`estimator.html`)
- Model machine learning dalam format MLflow

Artefak tersebut dapat diunduh melalui tab **Actions → Artifacts** pada repository GitHub.

---

## 🐳 Docker Hub Repository

Model machine learning yang telah dilatih dibangun menjadi Docker Image dan diunggah ke Docker Hub.

🔗 **Tautan Docker Hub Repository:**  
https://hub.docker.com/r/nikitanikita04/telco-churn-mlflow

Docker Image ini berisi model hasil training dan siap digunakan untuk keperluan deployment.

---

## 🧪 Dataset

Dataset yang digunakan pada proyek ini adalah:
- **Telco Customer Churn Dataset**  
  File: `Telco-Customer-Churn_preprocessing.csv`  
Dataset telah melalui tahap preprocessing sebelum digunakan dalam proses training.

