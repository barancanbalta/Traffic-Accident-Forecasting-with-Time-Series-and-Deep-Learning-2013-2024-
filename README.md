# Traffic Accident Forecasting (2013–2024)

This project applies time series modeling and deep learning to forecast monthly traffic accidents in Turkey using open data from TÜİK.

## 🔍 Goal
To compare classical time series models (SARIMA, Holt-Winters) with deep learning approaches (ANN, CNN) and evaluate their forecasting performance.

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
