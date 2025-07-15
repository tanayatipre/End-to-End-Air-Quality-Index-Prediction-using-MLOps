# End-To-End-Machine-Learning-Project-with-ML-Flow

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


# How to run?

### STEPS:

Clone the repository

```bash
https://github.com/tanayatipre/End-to-End-Machine-Learning-Project-with-MLFlow
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n MLProject python=3.10 -y
```

```bash
conda activate MLProject

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up your local host and port
```

## MLFlow

[Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/tanayatipre8/End-to-End-Machine-Learning-Project-with-MLFlow
MLFLOW_TRACKING_USERNAME=tanayatipre8
MLFLOW_TRACKING_PASSWORD=e65a37b4384f5f8f1cd181fc2223507eb0c58591 \
python script.py

Run this to export as env variables:

```bash
$env:MLFLOW_TRACKING_URI="https://dagshub.com/tanayatipre8/End-to-End-Machine-Learning-Project-with-MLFlow.mlflow"
$env:MLFLOW_TRACKING_USERNAME="tanayatipre8"
$env:MLFLOW_TRACKING_PASSWORD="3ad5427e8de276a639b9b41c06c13e42462c68fe"
```