import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

# Função para carregar e preparar os dados
def prepare_data(data_path, window_size):
    data = pd.read_csv(data_path, parse_dates=['time'])
    data = data.drop(columns=['bid_o', 'bid_h', 'bid_l', 'bid_c', 'ask_o', 'ask_h', 'ask_l', 'ask_c', 'mid_o', 'mid_h', 'mid_l', 'volume'])
    data = data[data['time'] > '2023-01-01'] 
    data['Target'] = (data['mid_c'].shift(-1) > data['mid_c']).astype(int)
    data.dropna(inplace=True)

    features = data.drop(columns=['time', 'Target'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return create_sequences(features_scaled, data['Target'].values, window_size), scaler

# Função para criar sequências de janelas deslizantes
def create_sequences(features, target, window_size):
    X = []
    y = []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

# Função para fazer predições
def predict(data_path, model_path, window_size):
    # Preparar os dados
    (X, y), scaler = prepare_data(data_path, window_size)
    
    # Carregar o modelo
    model = tf.keras.models.load_model(model_path)
    
    # Fazer a predição
    predictions = model.predict(X)
    
    return predictions

# Exemplo de uso
data_path = "C:\\Users\\heito\\Documents\\eps\\newData\\new_dataUSD_CAD_H1.csv"
model_path = 'model.h5'
window_size = 10

predictions = predict(data_path, model_path, window_size)
print(predictions[-1])
