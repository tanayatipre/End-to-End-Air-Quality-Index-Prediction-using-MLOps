## End-to-End Air Quality Index Prediction using MLOps

![Air Pollution](assets/AQI_Predictor.png)

This project builds an end-to-end machine learning pipeline for predicting the Air Quality Index (AQI), integrated with MLOps practices for automation, tracking, and deployment.

---

### Project Overview

The goal is to predict AQI based on atmospheric pollutant levels using an automated ML pipeline governed by MLOps practices. The solution spans data collection to live model deployment.

---

### Dataset

The dataset is sourced from the **Central Pollution Control Board (CPCB)**: [https://cpcb.nic.in/](https://cpcb.nic.in/).  
It contains pollutant data such as PM2.5, PM10, NO, NO2, CO, and O3 across various Indian cities.

---

### Implemented ML Pipeline Stages

1. Data Ingestion  
2. Data Validation  
3. Data Transformation & Feature Engineering  
4. Model Training & Hyperparameter Tuning  
5. Model Evaluation  
6. **Model Deployment** as a real-time prediction web service using Flask & Docker

---

### MLOps Stack

| Tool      | Purpose                   |
|-----------|---------------------------|
| 🧪 **MLflow**      | Experiment tracking         |
| 🐙 **GitHub Actions** | CI/CD automation            |
| 🐳 **Docker**        | Containerization            |
| ☁️ **AWS ECR**       | Docker image hosting        |
| 🌐 **DagsHub**       | Remote MLflow tracking      |

---

### Features

- Web interface to input pollutant values
- Real-time AQI prediction with health category (Good, Moderate, Poor, etc.)
- ML pipeline versioned and reproducible using MLOps

---

### Project Structure

```
.
├── src/MLProject/       # Core ML code
│   ├── components/      # Pipeline steps
│   ├── pipeline/        # Pipeline orchestration
│   └── utils/           # Helper functions
├── config/              # Configuration files
├── research/            # Jupyter notebooks
├── templates/           # Flask HTML templates
├── static/              # Static web assets
├── main.py              # Runs the ML pipeline
├── app.py               # Flask app for prediction
├── download_ml_artifacts.py
├── Dockerfile
├── .github/workflows/   # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

### Run the App

```bash
python main.py         # Run training pipeline
python app.py          # Launch Flask prediction app
```

---

