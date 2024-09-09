import pandas as pd
import numpy as np
from arch import arch_model
from models.trade_decision import TradeDecision

from technicals.indicators import BollingerBands

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


from api.oanda_api import OandaApi
from models.trade_settings import TradeSettings
import constants.defs as defs
from collections import defaultdict
import random

ADDROWS = 20

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

# Parâmetros do Q-Learning
Q_TABLE = defaultdict(lambda: defaultdict(float))
ALPHA = 0.1  # Taxa de aprendizado
GAMMA = 0.9  # Fator de desconto
EPSILON = 0.1  # Taxa de exploração

def apply_signal(row, trade_settings: TradeSettings):

    if row.SPREAD <= trade_settings.maxspread and row.GAIN >= trade_settings.mingain:
        if row.mid_c > row.BB_UP and row.mid_o < row.BB_UP:
            return defs.SELL
        elif row.mid_c < row.BB_LW and row.mid_o > row.BB_LW:
            return defs.BUY
    return defs.NONE

def apply_SL(row, trade_settings: TradeSettings):
    if row.SIGNAL == defs.BUY:
        return row.mid_c - (row.GAIN / trade_settings.riskreward)
    elif row.SIGNAL == defs.SELL:
        return row.mid_c + (row.GAIN / trade_settings.riskreward)
    return 0.0


def apply_TP(row):
    
    if row.SIGNAL == defs.BUY:
        return row.mid_c + row.GAIN
    elif row.SIGNAL == defs.SELL:
        return row.mid_c - row.GAIN
    return 0.0


def process_candles(df: pd.DataFrame, pair, trade_settings: TradeSettings, log_message):

    df.reset_index(drop=True, inplace=True)
    df['PAIR'] = pair
    df['SPREAD'] = df.ask_c - df.bid_c

    # Cálculo das Bollinger Bands
    df = BollingerBands(df, trade_settings.n_ma, trade_settings.n_std)
    df['GAIN'] = abs(df.mid_c - df.BB_MA)
    df['SIGNAL'] = df.apply(apply_signal, axis=1, trade_settings=trade_settings)
    df['TP'] = df.apply(apply_TP, axis=1)
    df['SL'] = df.apply(apply_SL, axis=1, trade_settings=trade_settings)
    df['LOSS'] = abs(df.mid_c - df.SL)

    log_cols = ['PAIR', 'time', 'mid_c', 'mid_o', 'SL', 'TP', 'SPREAD', 'GAIN', 'LOSS', 'SIGNAL']
    log_message(f"process_candles:\n{df[log_cols].tail()}", pair)

    return df[log_cols].iloc[-1]


def fetch_candles(pair, row_count, candle_time, granularity,
                    api: OandaApi, log_message):

    df = api.get_candles_df(pair, count=row_count, granularity=granularity)

    if df is None or df.shape[0] == 0:
        log_message("tech_manager fetch_candles failed to get candles", pair)
        return None
    
    if df.iloc[-1].time != candle_time:
        log_message(f"tech_manager fetch_candles {df.iloc[-1].time} not correct", pair)
        return None

    return df

def fit_garch(returns):
    """
    Ajusta um modelo GARCH(1, 1) para os retornos e retorna a previsão de volatilidade.
    """
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fitted = model.fit(disp="off")
    forecast = model_fitted.forecast(horizon=1)
    return forecast.variance.iloc[-1, 0]  # Volatilidade prevista

def optimize_portfolio(mean_return, volatility):
    """
    Simula a otimização de portfólio para um único ativo com base no retorno ajustado pelo risco.
    """
    risk_adjusted_return = mean_return / volatility
    return max(0, risk_adjusted_return)  # Peso não negativo

def choose_action(state):
    """
    Escolhe uma ação com base na política epsilon-greedy.
    """
    if random.uniform(0, 1) < EPSILON:
        return random.choice([defs.BUY, defs.SELL, defs.HOLD])
    return max(Q_TABLE[state], key=Q_TABLE[state].get, default=defs.HOLD)

def update_q_table(state, action, reward, next_state):
    """
    Atualiza a tabela Q com base na fórmula Q-learning.
    """
    best_next_action = max(Q_TABLE[next_state], key=Q_TABLE[next_state].get, default=defs.HOLD)
    td_target = reward + GAMMA * Q_TABLE[next_state][best_next_action]
    td_error = td_target - Q_TABLE[state][action]
    Q_TABLE[state][action] += ALPHA * td_error

def get_trade_decision(candle_time, pair, granularity, api: OandaApi, 
                            trade_settings: TradeSettings, log_message):


    max_rows = trade_settings.n_ma + ADDROWS

    log_message(f"tech_manager: max_rows:{max_rows} candle_time:{candle_time} granularity:{granularity}", pair)

    df = fetch_candles(pair, max_rows, candle_time,  granularity, api, log_message)

    if df is not None:
        # Processamento das candles e indicadores técnicos
        last_row = process_candles(df, pair, trade_settings, log_message)

        # Calcular retornos logarítmicos
        prices = df['mid_c']
        returns = np.log(prices / prices.shift(1)).dropna()

        # Modelo GARCH para prever a volatilidade
        predicted_volatility = fit_garch(returns)
        log_message(f"Volatilidade prevista para {pair}: {predicted_volatility}", pair)

        # Calcular retorno médio
        mean_return = returns.mean()

        # Otimização de portfólio com base no retorno médio e volatilidade
        optimal_weight = optimize_portfolio(mean_return, predicted_volatility)
        log_message(f"Peso ótimo calculado para {pair}: {optimal_weight}", pair)

        # Estado atual do ambiente (pode incluir outros parâmetros conforme necessário)
        state = (last_row.SIGNAL, predicted_volatility)

        # Escolher ação com base no Q-learning
        action = choose_action(state)
        log_message(f"Ação escolhida para {pair} com Q-learning: {action}", pair)

        # Aplicar a ação escolhida
        last_row.SIGNAL = action

        # Recompensa (por exemplo, pode ser baseada em retorno previsto, ajuste conforme necessário)
        reward = last_row.GAIN if action == defs.BUY else -last_row.LOSS

        # Atualizar a tabela Q com a recompensa e o próximo estado
        update_q_table(state, action, reward, state)  # Próximo estado é o mesmo aqui para simplificação

        return TradeDecision(last_row)

    return None


