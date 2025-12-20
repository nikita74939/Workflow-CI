# Workflow-CI

![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-Project-orange)

Repository ini dibuat untuk memenuhi **Kriteria 3: Membuat Workflow CI**. Proyek ini mengimplementasikan pipeline *Continuous Integration* (CI) penuh untuk Machine Learning menggunakan **MLflow Project**, **GitHub Actions**, **Google Drive**, dan **Docker Hub**.

---

## ⚙️ Alur Workflow CI/CD

Setiap kali ada perubahan (*push*) ke branch `main`, GitHub Actions akan menjalankan serangkaian proses otomatis:

1. **Environment Setup**: Menginstal Python dan dependensi dari `MLProject/conda.yaml`.
2. **Model Training**: Menjalankan `mlflow run` untuk melatih model Logistic Regression secara otomatis.
3. **Artifact Archiving**: Mengompres hasil eksperimen (`mlruns`) menjadi file `.zip`.
4. **Storage Backup (Advance)**:
* Mengunggah artefak ke **GitHub Actions Artifacts**.
* Mengunggah cadangan artefak ke **Google Drive** secara otomatis.


5. **Deployment**:
* Membangun Docker Image dari model yang baru dilatih (`mlflow models build-docker`).
* Mengunggah (*push*) image tersebut ke **Docker Hub**.



---

## 🐳 Docker Hub Repository

Model machine learning yang telah dilatih otomatis dibungkus menjadi Docker Image dan siap untuk dideploy.

🔗 **Tautan Docker Hub:** **[hub.docker.com/r/nikitanikita04/telco-churn-mlflow](https://hub.docker.com/r/nikitanikita04/telco-churn-mlflow)**

*Tag image: `latest` akan selalu berisi model hasil training terbaru.*

---

## ☁️ Integrasi Google Drive

Sebagai bagian dari kriteria *Advance*, setiap hasil run (termasuk model dan metrik) otomatis dicadangkan ke folder Google Drive menggunakan Service Account.

* **Format File:** `telco_churn_run_<run_id>.zip`
* **Lokasi:** Disimpan di folder khusus Google Drive yang telah dikonfigurasi via GitHub Secrets.

---

## 📊 Artefak & Hasil

Pipeline ini menghasilkan berbagai artefak evaluasi model yang tersimpan di dalam arsip:

* **Model**: File model MLflow (`model.pkl`, `MLmodel`, `conda.yaml`).
* **Visualisasi**: Confusion Matrix (`training_confusion_matrix.png`).
* **Data**: Koefisien Regresi (`lr_coefficients.csv`).
* **Metrik**: Ringkasan performa (`metric_info.json`, `estimator.html`).

---

## 🛠️ Cara Menjalankan Lokal

Jika ingin menjalankan project ini di komputer lokal tanpa GitHub Actions:

1. Pastikan **MLflow** sudah terinstal.
2. Masuk ke folder project:
```bash
cd MLProject

```


3. Jalankan perintah run:
```bash
mlflow run . --env-manager=local

```



---

*Dibuat oleh Nikita.*

