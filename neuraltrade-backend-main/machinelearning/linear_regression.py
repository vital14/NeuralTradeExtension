import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objs as go
import plotly.express as px

# Load and preprocess your data
data = pd.read_csv("../data/hist_1y_1h.csv")

# Preprocessing
X = data[['Open', 'High', 'Low', 'Volume']].values
y = data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer (1 neuron for regression)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print("Mean Squared Error:", mse)

# Get predictions on the test data
y_pred = model.predict(X_test_scaled)

# Create a DataFrame for easier visualization
results = pd.DataFrame({'Actual Close': y_test, 'Predicted Close': y_pred.flatten()})

# Create a scatter plot
fig = px.scatter(results, x='Actual Close', y='Predicted Close', title='Actual vs Predicted Close Prices')
fig.update_traces(marker=dict(size=8, opacity=0.6))

# Add a diagonal line for reference
fig.add_trace(go.Scatter(x=results['Actual Close'], y=results['Actual Close'], mode='lines', name='Ideal'))

# Add axis labels
fig.update_layout(xaxis_title='Actual Close Price', yaxis_title='Predicted Close Price')

# Show the plot
fig.show()
