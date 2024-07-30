import yfinance as yf
import pandas as pd
import numpy as np
from pymongo import MongoClient
from indicatorsFunction import *

def insert_data_into_mongodb(collection_name, data):
    client = MongoClient('mongodb+srv://neuraltrade:eps-neuraltrade@neuraltrade.1ykvwjf.mongodb.net/?retryWrites=true&w=majority&appName=NeuralTrade')
    db = client['NeuralTradeDB']
    collection = db[collection_name]
    collection.insert_many(data)
    print(f"Dados de {collection_name} inseridos com sucesso no MongoDB.")

currencies = ['USDCAD', 'USDJPY']

for currency in currencies:
    asset = yf.Ticker(currency + '=X')
    data_60days_5m = asset.history(period="60d", interval="5m")
    data_1y_1h = asset.history(period="1y", interval="1h")
    data_5y_1d = asset.history(period="5y", interval="1d")
    data_10y_5d = asset.history(period="10y", interval="5d")

    datasets = {
        '60d_5m': data_60days_5m,
        '1y_1h': data_1y_1h,
        '5y_1d': data_5y_1d,
        '10y_5d': data_10y_5d
    }

    for period, data in datasets.items():
        if not isinstance(data.index, pd.DatetimeIndex):
            data['Datetime'] = data.index

        data['EMA_12'] = EMA(data['Close'], 12)
        data['EMA_26'] = EMA(data['Close'], 26)
        data['SMA_50'] = SMA(data['Close'], 50)
        data['WMA_50'] = WMA(data['Close'], 50)
        data['HMA_50'] = HMA(data['Close'], 50)
        data['MACD'] = MACD(data['Close'])
        data['TypicalPrice'] = TypicalPrice(data['High'], data['Low'], data['Close'])
        data['CCI'] = CCI(data['TypicalPrice'])
        data['StochasticOscillator'] = StochasticOscillator(data['Close'], data['High'], data['Low'])
        data['RSI'] = RSI(data['Close'])
        data['ROC_12'] = ROC(data['Close'], 12)
        data['PPO'] = PPO(data['Close'])
        data['KST'] = KST(data['Close'])
        data['BollingerBandUp'] = BollingerBandUp(data['TypicalPrice'])
        data['BollingerBandMiddle'] = BollingerBandMiddle(data['TypicalPrice'])
        data['BollingerBandDown'] = BollingerBandDown(data['TypicalPrice'])

        data.drop(columns=['Volume', 'Dividends', 'Stock Splits'], inplace=True)

        data.reset_index(inplace=True)

        data_dict = data.to_dict(orient='records')

        insert_data_into_mongodb(f"{currency}_{period}", data_dict)
