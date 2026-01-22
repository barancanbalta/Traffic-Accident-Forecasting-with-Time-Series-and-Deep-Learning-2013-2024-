# Traffic Accident Forecasting with Time Series and Deep Learning (2013-2024)

A comprehensive forecasting analysis comparing classical time series models (SARIMA, Holt-Winters) with deep learning approaches (ANN, CNN) for predicting monthly traffic accidents in Turkey using TÜİK open data.

## Overview

This project demonstrates the application of both traditional statistical methods and modern deep learning techniques for time series forecasting. The analysis spans 11 years of monthly traffic accident data to identify the most effective prediction model.

## 🚀 Best Performance (ANN Model)
- **R²**: 0.9315  
- **MAPE**: 3.62%  
- **RMSE**: 947.92  
- **Best Parameters**:  
  - `look_back = 24`  
  - `neurons = 32-16`  
  - `activation = relu`  
  - `batch_size = 32`  
  - `learning_rate = 0.001`  
  - `scaler = MinMaxScaler`  

## 📁 Project Structure

- `data/2024_eklenmiş.xlsx` – Historical accident data (2013–2024)
- `src/ANN_BEST.py` – Optimized ANN model with grid search
- `outputs/` – Prediction plots and results

## 📊 Sample Output

![Forecast Plot](outputs/best_model_plot.png)

## 🛠 Tools & Libraries

- Python, pandas, NumPy, scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- ExcelWriter, datetime

## 📌 Dataset Source
Data obtained from [TÜİK – Türkiye İstatistik Kurumu](https://www.tuik.gov.tr/)

---

## ✅ How to Run

```bash
pip install -r requirements.txt
python src/ANN_BEST.py
```
