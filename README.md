# Energy Consumption Prediction — Feedforward Neural Network

A feedforward neural network (FNN) built with TensorFlow/Keras to predict
energy consumption from environmental sensor data.

## Problem Statement
Predict energy consumption (kWh) from 4 environmental inputs:
temperature, humidity, wind speed, and solar irradiance.

## Model Architecture
Input (4 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)

## Dataset
Synthetically generated using NumPy (seed=42). 1000 data points, 80/20 split.

| Feature | Range |
|---------|-------|
| Temperature | 15–45 °C |
| Humidity | 20–90% |
| Wind Speed | 0–20 km/h |
| Solar Irradiance | 100–1000 W/m² |

## Visualizations

### Training vs Validation Loss
![Training Loss](training_loss.png)

### Predicted vs Actual Energy Consumption
![Predictions vs Actual](predictions_vs_actual.png)

### Feature Distributions
![Feature Distributions](feature_distributions.png)

## Key Findings
- Model converges cleanly with no overfitting across 50 epochs
- Validation loss closely tracks training loss — strong generalisation
- Predictions cluster tightly near the perfect prediction line
- StandardScaler normalisation was critical for stable training

## Libraries Used
TensorFlow/Keras · scikit-learn · pandas · numpy · matplotlib

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Energy_Consumption_FNN.ipynb
```
