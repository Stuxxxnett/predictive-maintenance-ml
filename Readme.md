# Predictive Maintenance for Machines

ML system to predict equipment failure from sensor data (AI4I 2020 dataset), served via FastAPI.

## Features
- Trained RandomForestClassifier on sensor readings (temperature, torque, rotational speed, tool wear)
- Handles class imbalance with `class_weight="balanced"`
- ROC-AUC: 0.97
- REST API for real-time predictions

## Project Structure