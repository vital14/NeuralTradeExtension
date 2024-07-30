import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
from datetime import datetime, timedelta
import schedule
import time
from data.processData import *

# Função para calcular o número de velas
def calculate_count(granularity, period_days):
    if granularity == "H1":
        candles_per_day = 24
    elif granularity == "M1":
        candles_per_day = 24 * 60
    elif granularity == "D":
        candles_per_day = 1
    elif granularity == "W":
        candles_per_day = 1 / 7
    else:
        raise ValueError("Granularidade não suportada")
    count = int(candles_per_day * period_days)
    return count

# Função para carregar e preparar os dados
def prepare_data(data, window_size):

    data['EMA_12'] = EMA(data['mid_c'], 12)
    data['EMA_26'] = EMA(data['mid_c'], 26)
    data['SMA_20'] = SMA(data['mid_c'], 20)
    data['WMA_20'] = WMA(data['mid_c'], 20)
    data['HMA_20'] = HMA(data['mid_c'], 20)
    data['MACD'] = MACD(data['mid_c'])
    data['TypicalPrice'] = TypicalPrice(data['mid_h'], data['mid_l'], data['mid_c'])
    data['CCI'] = CCI(data['TypicalPrice'])
    data['StochasticOscillator'] = StochasticOscillator(data['mid_c'], data['mid_h'], data['mid_l'])
    data['RSI'] = RSI(data['mid_c'])
    data['ROC'] = ROC(data['mid_c'], 20)
    data['PPO'] = PPO(data['mid_c'])
    data['KST'] = KST(data['mid_c'])
    data['BollingerBandUp'] = BollingerBandUp(data['TypicalPrice'])
    data['BollingerBandMiddle'] = BollingerBandMiddle(data['TypicalPrice'])
    data['BollingerBandDown'] = BollingerBandDown(data['TypicalPrice'])

    data = data.drop(columns=['mid_o', 'mid_h', 'mid_l'])
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

# Função para obter dados do OANDA
def get_oanda_data(instrument, granularity, count, access_token):
    client = oandapyV20.API(access_token=access_token)
    max_count = 4992  # Limite máximo permitido pela API do OANDA
    params = {
        "granularity": granularity,
        "count": max_count
    }
    all_data = []
    while count > 0:
        if count < max_count:
            params["count"] = count
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response['candles']
        for candle in candles:
            mid = candle['mid']
            all_data.append({
                'time': candle['time'],
                'mid_o': float(mid['o']),
                'mid_h': float(mid['h']),
                'mid_l': float(mid['l']),
                'mid_c': float(mid['c'])
            })
        count -= max_count
        if len(candles) > 0:
            last_time = datetime.strptime(candles[-1]['time'], '%Y-%m-%dT%H:%M:%S.%f000Z')
            params["from"] = (last_time + timedelta(seconds=1)).isoformat()
    return pd.DataFrame(all_data)

# Função para fazer predições
def predict(instrument, granularity, period_days, model_path, window_size, access_token):
    count = calculate_count(granularity, period_days)
    data = get_oanda_data(instrument, granularity, count, access_token)
    (X, y), scaler = prepare_data(data, window_size)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X)
    return predictions

# Função para executar a ação de trade
def trade_action(action, prediction):
    client = oandapyV20.API(access_token=access_token)
    account_id = "101-001-29394955-001"
    if action == "buy":
        units = int(1000*prediction[0])
    elif action == "sell":
        units = int(-1000*(1-prediction[0]))
    order = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = oandapyV20.endpoints.orders.OrderCreate(account_id, data=order)
    client.request(r)
    print(f"Trade executed: {action} {units} units")

# Função para executar predições e realizar trades
def execute_trades():
    print("Executing trades...")
    predictions = predict(instrument, granularity, period_days, model_path, window_size, access_token)
    last_prediction = predictions[-1]
    print(f"Last prediction: {last_prediction}")
    if last_prediction > 0.6:  # Exemplo: se a previsão for maior que 0.5, comprar
        trade_action("buy", last_prediction)
    elif last_prediction < 0.4:  # Caso contrário, vender
        trade_action("sell", last_prediction)
    else:
        print("No trade executed")

# Parâmetros
instrument = "USD_CAD"
granularity = "H1"
period_days = 200
model_path = '/home/hmsb/neuraltrade-backend/machinelearning/model.h5'
window_size = 10
access_token = '6bd4adea352ae6cea5125b8147ce1674-597367a27602680acd278ce52193d2bc'

# Agendar a execução para cada hora
schedule.every().hour.do(execute_trades)

# Agendar a execução para cada minuto
# schedule.every().minute.do(execute_trades)

# Loop para manter o script em execução e verificar o agendamento
while True:
    schedule.run_pending()
    time.sleep(1)
