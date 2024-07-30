import pandas as pd
import plotly.graph_objs as go

file_paths = {
    '1h': './hist_1y_1h.csv',
    '1d': './hist_5y_1d.csv',
    '5d': './hist_10y_5d.csv',
    '5minutes': './hist_60d_5m.csv'
}

fig = go.Figure()

buttons = []
for interval, file_path in file_paths.items():
    buttons.append(
        dict(
            args=[{'x': [pd.read_csv(file_path)['Datetime']], 'y': [pd.read_csv(file_path)['Close']],
                   'name': ['Close']}],
            label=interval,
            method='update'
        )
    )

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            direction='right',
            active=0,
            x=0.1,
            y=1.1,
            buttons=buttons
        )
    ]
)

initial_file = list(file_paths.keys())[0]
df = pd.read_csv(file_paths[initial_file])
fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], mode='lines', name='Close'))

fig.update_layout(
    title=f'Stock Price ({initial_file})',
    xaxis_title='Datetime',
    yaxis_title='Price',
    xaxis=dict(tickangle=-45),
)

fig.show()
