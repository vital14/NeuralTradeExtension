import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def avalia_modelo(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Acurácia: {accuracy}")

    # Previsões
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    return y_pred_classes

def create_predicted_table(model, X_train, X_test, data, window_size):
    X_all = np.concatenate((X_train, X_test), axis=0)

    pred_all = model.predict(X_all)
    y_pred = model.predict(X_test)

    # y_pred_classes_all = (pred_all > 0.5).astype(int)
    # y_pred_classes = (y_pred > 0.5).astype(int)

    threshold_sell = 0.3
    threshold_buy = 0.7

    # y_pred_classes if < 0.4 then -1 else if > 0.6 then 1 else 0
    y_pred_classes_all = np.where(pred_all < threshold_sell, -1, np.where(pred_all > threshold_buy, 1, 0))
    y_pred_classes = np.where(y_pred < threshold_sell, -1, np.where(y_pred > threshold_buy, 1, 0))

    print(len(y_pred_classes_all))
    print(len(y_pred_classes))

    # Create a DataFrame for the predictions_all
    predicted_dates_all = data.iloc[window_size:window_size + len(y_pred_classes_all)].index
    predictions_all = pd.DataFrame(data={'Datetime': predicted_dates_all, 'Predicted': y_pred_classes_all.flatten(), 'accuracy': pred_all.flatten()})
    predictions_all.set_index('Datetime', inplace=True)

    # Create a DataFrame for the predictions_test
    predicted_dates_test = data.iloc[window_size:window_size + len(y_pred_classes)].index
    predictions_test = pd.DataFrame(data={'Datetime': predicted_dates_test, 'Predicted': y_pred_classes.flatten(), 'accuracy': y_pred.flatten()})
    predictions_test.set_index('Datetime', inplace=True)

    return predictions_all, predictions_test

def plot_predictions(predictions_all, predictions_test):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    predictions_all['accuracy'].groupby(pd.cut(predictions_all['accuracy'], np.arange(0, 1.1, 0.1)).astype(str)).count().plot(kind='bar', ax=axs[0]).set_title('All data')
    predictions_test['accuracy'].groupby(pd.cut(predictions_test['accuracy'], np.arange(0, 1.1, 0.1)).astype(str)).count().plot(kind='bar', ax=axs[1]).set_title('Test data')

def plot_predictions_and_prices(data, predictions_all):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(predictions_all.index, predictions_all['Predicted'], label='Predicted', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('USDCAD Close Prices and Predicted Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_returns(predictions_all, predictions_test, data):

    # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
    predictions_all['Signal'] = predictions_all['Predicted']

    # Calculate the returns
    data['Returns'] = data['Close'].pct_change()

    # Align predictions_all with returns
    predictions_all = predictions_all.join(data['Returns'], how='left')

    # Calculate strategy returns
    predictions_all['Strategy_Returns'] = predictions_all['Signal'].shift(1) * predictions_all['Returns']

    # Calculate cumulative returns
    predictions_all['Cumulative_Strategy_Returns'] = (1 + predictions_all['Strategy_Returns']).cumprod() - 1
    predictions_all['Cumulative_Market_Returns'] = (1 + predictions_all['Returns']).cumprod() - 1


    predictions_test['Signal'] = predictions_test['Predicted']

    # Calculate the returns
    data['Returns'] = data['Close'].pct_change()

    # Align predictions_test with returns
    predictions_test = predictions_test.join(data['Returns'], how='left')

    # Calculate strategy returns
    predictions_test['Strategy_Returns'] = predictions_test['Signal'].shift(1) * predictions_test['Returns']

    # Calculate cumulative returns
    predictions_test['Cumulative_Strategy_Returns'] = (1 + predictions_test['Strategy_Returns']).cumprod() - 1
    predictions_test['Cumulative_Market_Returns'] = (1 + predictions_test['Returns']).cumprod() - 1

    return predictions_all, predictions_test

def plot_returns(predictions_all, predictions_test):
    # Create the figure and axes for the subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the first graph on ax1
    # ax1.plot(x_values, y_values1, color='blue')
    ax1.plot(predictions_all.index, predictions_all['Cumulative_Strategy_Returns'], label='Strategy Returns')
    ax1.plot(predictions_all.index, predictions_all['Cumulative_Market_Returns'], label='Market Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Cumulative Strategy Returns vs Market Returns (All data)')
    ax1.legend()
    ax1.grid(True)


    # Plot the second graph on ax2
    ax2.plot(predictions_test.index, predictions_test['Cumulative_Strategy_Returns'], label='Strategy Returns')
    ax2.plot(predictions_test.index, predictions_test['Cumulative_Market_Returns'], label='Market Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Returns')
    ax2.set_title('Cumulative Strategy Returns vs Market Returns (Test data)')
    ax2.legend()
    ax2.grid(True)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def calculate_return_difference(predictions_all, predictions_test):
    strategy_return_all = predictions_all['Cumulative_Strategy_Returns'].iloc[-1]
    market_return_all = predictions_all['Cumulative_Market_Returns'].iloc[-1]
    
    strategy_return_test = predictions_test['Cumulative_Strategy_Returns'].iloc[-1]
    market_return_test = predictions_test['Cumulative_Market_Returns'].iloc[-1]

    # Diferença percentual dos retornos cumulativos
    return_diff_all = (strategy_return_all - market_return_all) * 100
    return_diff_test = (strategy_return_test - market_return_test) * 100
    
    return return_diff_all, return_diff_test