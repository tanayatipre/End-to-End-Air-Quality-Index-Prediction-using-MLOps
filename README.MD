# End-to-End Air Quality Index Prediction using MLOps

This project establishes a machine learning pipeline for predicting the Air Quality Index (AQI). The focus is on building an automated MLOps framework that covers the entire lifecycle from data acquisition to model deployment and serving.

---

## **Project Overview**

The objective of this project is to predict the Air Quality Index (AQI) based on various atmospheric pollutant concentrations. The implemented solution is designed as an end-to-end machine learning pipeline, incorporating MLOps principles for automation.

---

## **Dataset Information**

The data used for this predictive model is air quality information obtained from the **Central Pollution Control Board (CPCB)** in India. The original source for this data is [https://cpcb.nic.in/](https://cpcb.nic.in/).

The dataset includes daily measurements of pollutants such as PM2.5, PM10, NO, NO2, CO, and O3, correlated with their corresponding AQI values across different Indian cities.

---

## **Implemented ML Pipeline Stages**

The project's machine learning pipeline is structured into several automated and sequential stages:

1.  **Data Ingestion:** Raw air quality data is acquired from the specified source.
2.  **Data Validation:** The ingested data undergoes validation against a predefined schema to ensure integrity and consistency before further processing.
3.  **Data Transformation & Feature Engineering:** This stage processes the raw data into a format suitable for model training. Operations include:
    * Handling of missing values through imputation.
    * Application of transformations (e.g., log transformation) to numerical pollutant features.
    * Scaling of numerical features.
    * Derivation of new temporal features (Year, Month, Day, Day of Week, Weekend indicator) from date information.
    * Encoding of categorical features.
4.  **Model Training & Hyperparameter Tuning:** A **CatBoost Regressor** is employed for the prediction task. Model performance is optimized through integrated **RandomizedSearchCV** for hyperparameter tuning. This process systematically identifies optimal parameter combinations for the CatBoost model.
5.  **Model Evaluation:** The trained model's performance is assessed using unseen test data. Standard regression metrics such as R-squared (R²), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are computed. The fine-tuned model consistently achieves a high R² score (typically above 0.91), indicating strong predictive capability.

---

## **MLOps Tooling**

The project leverages several tools:

* **MLflow:** Used for experiment tracking, allowing for the logging and versioning of all model training parameters, performance metrics, and generated artifacts (including the trained model and data preprocessor).
* **DagsHub:** Serves as the remote MLflow Tracking Server. It centralizes all experiment runs, providing a web-based platform for visualization, comparison, and collaboration on model development.
* **GitHub Actions:** Automates the Continuous Integration/Continuous Deployment (CI/CD) pipeline. 
* **Docker:** Facilitates the containerization of the prediction application. 
* **AWS ECR (Elastic Container Registry):** Used for secure storage of Docker images built by the CI/CD pipeline, making them ready for deployment.
* **Flask:** Provides the lightweight web framework for the prediction API and user interface.

---

## **Model Deployment**

The trained and validated model is deployed as a live prediction service. The deployed Docker container dynamically downloads the specific pre-trained model and data preprocessor using MLflow artifacts. The service features:

* A web interface for user input of pollutant data and dates.
* **Server-side input validation** that checks if pollutant values and dates fall within expected, realistic ranges, providing immediate feedback for out-of-bounds inputs.
* Real-time AQI score prediction.
* Categorization of the predicted AQI into interpretative buckets (e.g., Good, Moderate, Severe).

---

## **Project Directory Structure**

The project is organized into the following logical components:

```
.
├── .github/workflows/       # GitHub Actions CI/CD workflow definitions
├── artifacts/               # Local repository for pipeline output artifacts
├── config/                  # Configuration files (e.g., config.yaml)
├── logs/                    # Runtime logs for pipeline executions
├── research/                # Jupyter notebooks for exploratory data analysis and initial experimentation
├── src/MLProject/           # Core Python source code for the ML application
│   ├── components/          # Modular implementations of individual pipeline steps
│   ├── config/              # Configuration loading and management
│   ├── constants/           # Global constants
│   ├── entity/              # Data classes for defining data structures and configurations
│   ├── pipeline/            # Orchestrates the sequence of ML pipeline stages
│   └── utils/               # General utility functions
├── static/                  # Web assets (CSS, images) for the Flask application
├── templates/               # HTML templates for the Flask web UI
├── .gitignore               # Specifies files/directories to be ignored by Git
├── app.py                   # Flask web application entry point
├── Dockerfile               # Docker image build instructions
├── download_ml_artifacts.py # Python script for downloading MLflow artifacts
├── entrypoint.sh            # Entrypoint script for the Docker container
├── main.py                  # Main script to execute the full ML training pipeline
├── params.yaml              # Parameter configurations for models and pipeline steps
├── README.md                # This document
├── requirements.txt         # Python package dependencies
└── schema.yaml              # Defines the expected data schema for validation
