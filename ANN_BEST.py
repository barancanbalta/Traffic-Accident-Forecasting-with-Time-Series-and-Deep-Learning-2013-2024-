import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def create_model(neurons1, neurons2, learning_rate, activation):
    model = Sequential([
        Dense(neurons1, input_dim=look_back, activation=activation),
        Dense(neurons2, activation=activation),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Grid search parametreleri
param_grid = {
    'look_back': [6, 12, 24],
    'neurons1': [32, 64, 128],
    'neurons2': [16, 32, 64],
    'batch_size': [8, 16, 32],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'activation': ['relu', 'tanh'],
    'scaler': [MinMaxScaler(), StandardScaler()]
}

# Veriyi oku
df = pd.read_excel("C:/mnt/data/2024_eklenmiş.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Eğitim ve test setlerini ayır
train_df = df[:'2023-12']
test_df = df['2024-01':]

# Sonuçları saklamak için liste
results = []

# Tüm parametre kombinasyonlarını oluştur
param_combinations = list(itertools.product(
    param_grid['look_back'],
    param_grid['neurons1'],
    param_grid['neurons2'],
    param_grid['batch_size'],
    param_grid['learning_rate'],
    param_grid['activation'],
    param_grid['scaler']
))

total_combinations = len(param_combinations)
print(f"Toplam {total_combinations} farklı kombinasyon denenecek.\n")

for idx, (look_back, neurons1, neurons2, batch_size, learning_rate, activation, scaler) in enumerate(param_combinations, 1):
    print(f"Kombinasyon {idx}/{total_combinations}")
    print(f"Parameters: look_back={look_back}, neurons1={neurons1}, neurons2={neurons2}, "
          f"batch_size={batch_size}, lr={learning_rate}, activation={activation}, scaler={type(scaler).__name__}")
    
    # Veriyi hazırla
    train_series = train_df['Accidents'].values.reshape(-1, 1)
    test_series = test_df['Accidents'].values.reshape(-1, 1)
    
    # Normalize et
    scaled_train = scaler.fit_transform(train_series)
    
    # Dataset oluştur
    X_train, y_train = create_dataset(scaled_train, look_back)
    
    # Model oluştur ve eğit
    model = create_model(neurons1, neurons2, learning_rate, activation)
    early_stopping = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Tahminler
    predictions = []
    input_seq = scaled_train[-look_back:]
    
    for _ in range(len(test_df)):
        input_reshaped = input_seq.reshape(1, look_back)
        pred = model.predict(input_reshaped, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.append(input_seq[1:], pred, axis=0)
    
    predicted = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Metrikler
    rmse = np.sqrt(mean_squared_error(test_series, predicted))
    mape = mean_absolute_percentage_error(test_series, predicted) * 100
    r2 = r2_score(test_series, predicted)
    
    # Sonuçları sakla
    results.append({
        'look_back': look_back,
        'neurons1': neurons1,
        'neurons2': neurons2,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'activation': activation,
        'scaler': type(scaler).__name__,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': predicted.flatten(),
        'model': model
    })
    
    print(f"R²: {r2:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}\n")

# En iyi modeli bul (R² değerine göre)
best_result = max(results, key=lambda x: x['r2'])

print("\nEn İyi Model Parametreleri:")
print(f"Look-back: {best_result['look_back']}")
print(f"Neurons: {best_result['neurons1']}-{best_result['neurons2']}")
print(f"Batch Size: {best_result['batch_size']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Activation: {best_result['activation']}")
print(f"Scaler: {best_result['scaler']}")
print(f"\nEn İyi Model Performansı:")
print(f"R²: {best_result['r2']:.4f}")
print(f"MAPE: {best_result['mape']:.2f}%")
print(f"RMSE: {best_result['rmse']:.2f}")

# Sonuçları DataFrame'e dönüştür
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model' and k != 'predictions'} 
                          for r in results])

# En iyi modelin tahminlerini DataFrame'e dönüştür
best_predictions_df = pd.DataFrame({
    'Month': test_df.index.strftime("%Y-%m"),
    'Actual': test_df['Accidents'].values,
    'Predicted': best_result['predictions'],
    'Error': np.abs(test_df['Accidents'].values - best_result['predictions']),
    'Error %': np.abs((test_df['Accidents'].values - best_result['predictions']) / test_df['Accidents'].values * 100)
})

# Sonuçları Excel'e kaydet
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f'model_results_{timestamp}.xlsx'
with pd.ExcelWriter(excel_filename) as writer:
    results_df.to_excel(writer, sheet_name='All_Results', index=False)
    best_predictions_df.to_excel(writer, sheet_name='Best_Model_Predictions', index=False)

print(f"\nSonuçlar '{excel_filename}' dosyasına kaydedildi.")

# En iyi modelin grafiğini çiz
plt.figure(figsize=(15, 6))

# Tahmin grafiği
plt.subplot(1, 2, 1)
plt.plot(best_predictions_df["Month"], best_predictions_df["Actual"], 
         label="Gerçek", marker="o", linewidth=2)
plt.plot(best_predictions_df["Month"], best_predictions_df["Predicted"], 
         label="Tahmin", marker="x", linewidth=2)
plt.xticks(rotation=45)
plt.title("En İyi Model - 2024 Tahminleri")
plt.xlabel("Aylar")
plt.ylabel("Kaza Sayısı")
plt.legend()
plt.grid(True)

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(best_predictions_df["Actual"], best_predictions_df["Predicted"])
plt.plot([min(test_df['Accidents']), max(test_df['Accidents'])], 
         [min(test_df['Accidents']), max(test_df['Accidents'])], 
         'r--', label='Perfect Prediction')
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.title("Gerçek vs Tahmin Karşılaştırması")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'best_model_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# En iyi model mimarisi
print("\nEn İyi Model Mimarisi:")
best_result['model'].summary() 