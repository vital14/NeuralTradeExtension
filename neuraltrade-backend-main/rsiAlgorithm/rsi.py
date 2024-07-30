import numpy as np

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gain = deltas * 0
    loss = gain.copy()

    gain[deltas > 0] = deltas[deltas > 0]
    loss[deltas < 0] = -deltas[deltas < 0]

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])

    for i in range(period, len(prices)):
        avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i]) / period
        avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i]) / period

    rs = np.zeros_like(prices)
    rs[period:] = avg_gain[period:] / avg_loss[period:]
    rsi = np.zeros_like(prices)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    # Lidar com casos especiais quando avg_loss é zero ou ambos avg_gain e avg_loss são zero
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 50
    
    return rsi

# Exemplo de uso:
historical_prices = [100, 102, 98, 105, 101, 99, 103, 107, 109, 110, 105, 108, 106, 104]
rsi_values = calculate_rsi(historical_prices)
print(rsi_values)
