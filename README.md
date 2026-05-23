# 🚀 MLOps-Core: End-to-End Automated ML Pipeline & Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-F37021?logo=scikit-learn)

A production-ready **Machine Learning Engineering & MLOps** portfolio project. This repository demonstrates how to transition from local Jupyter Notebooks to a fully automated, containerized, and deployable Machine Learning system.

## 💡 Overview

This project implements a continuous integration and continuous deployment (CI/CD) pipeline for a Machine Learning model (Breast Cancer Classification using LightGBM). Instead of a traditional UI, the model is served via a highly scalable REST API using **FastAPI** and packaged inside a **Docker** container.

Whenever new code or data is pushed to this repository, **GitHub Actions** automatically:
1. Installs dependencies.
2. Runs unit tests via `pytest`.
3. Retrains the Machine Learning model.
4. Generates a performance report (Accuracy, F1-Score) and a Confusion Matrix.
5. Posts the results directly to the GitHub Actions Job Summary.

---

## 🏗️ Architecture & Tech Stack

*   **Model Training:** `LightGBM`, `Scikit-Learn`, `Pandas`, `NumPy`
*   **Model Evaluation:** `Matplotlib`, `Seaborn`
*   **API & Deployment:** `FastAPI`, `Uvicorn`
*   **Containerization:** `Docker`
*   **CI/CD & Automation:** `GitHub Actions`, `Pytest`

---

## 📂 Project Structure

```text
MLOps-Core/
├── .github/workflows/
│   └── ml_pipeline.yml     # CI/CD Automation Pipeline
├── api/
│   └── main.py             # FastAPI REST endpoint
├── src/
│   └── train.py            # Model training and report generation script
├── tests/
│   ├── test_api.py         # Pytest for FastAPI endpoints
│   └── test_model.py       # Pytest for model training logic
├── models/                 # Saved model (.pkl) (Generated automatically)
├── reports/                # Performance metrics and plots (Generated automatically)
├── Dockerfile              # Instructions to containerize the API
└── requirements.txt        # Python dependencies
```

---

## ⚙️ How to Run Locally

### Option 1: Using Docker (Recommended)
If you have Docker installed, you can run the entire API in an isolated container without installing Python libraries on your machine.

```bash
# 1. Build the Docker Image
docker build -t mlops-core-api .

# 2. Run the Docker Container
docker run -d -p 8000:8000 mlops-core-api
```
*Once running, visit `http://localhost:8000/docs` to interact with the API via the automated Swagger UI.*

### Option 2: Using Standard Python
```bash
# 1. Clone the repository
git clone https://github.com/DtScntst1/MLOps-Core.git
cd MLOps-Core

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Train the model (generates models/model.pkl)
python src/train.py

# 4. Start the FastAPI server
uvicorn api.main:app --reload
```

---

## 🤖 CI/CD Automation in Action

Check out the **Actions** tab in this repository! You will see the automated workflow. If any test fails, the pipeline stops, preventing bad code or poor models from reaching production. If it passes, the latest model metrics are automatically generated and displayed.
