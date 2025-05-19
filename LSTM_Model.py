import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['country'] == "Tunisian Republic"]
    df['attackdate'] = pd.to_datetime(df['attackdate'])
    df.set_index('attackdate', inplace=True)
    df = df.sort_index()
    full_index = pd.date_range(start='2022-10-01', end='2023-12-31', freq='D')
    daily_attacks = df.groupby(df.index)['total_attacks'].sum()
    daily_attacks = daily_attacks.reindex(full_index, fill_value=0)
    daily_attacks = daily_attacks.rolling(window=3).mean().fillna(0)
    daily_attacks.index.name = 'attackdate'
    return daily_attacks

def create_sequences(data, seq_length=14):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_predictions(train_index, test_index, train_true, test_true, test_pred, country_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_index, y=train_true, name='Training Data', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=test_index, y=test_true, name='Actual Values', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=test_index, y=test_pred, name='Predicted Values', line=dict(color='red', width=2)))
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)
    r2 = r2_score(test_true, test_pred)
    actual_trend = np.gradient(test_true)
    pred_trend = np.gradient(test_pred)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]
    fig.update_layout(
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (LSTM)',
        xaxis_title='Date',
        yaxis_title='Total Attacks',
        width=1200,
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    fig.add_annotation(
        text=f'RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R²: {r2:.4f}<br>Trend Correlation: {trend_corr:.4f}',
        xref='paper',
        yref='paper',
        x=0.99,
        y=0.85,
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        align='right',
        xanchor='right'
    )
    fig.show()

def main():
    country_name = "Tunisian Republic"
    series = load_and_prepare_data('news_all_cleaned_processed.csv')
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    seq_length = 14
    X, y = create_sequences(scaled_values, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = build_lstm_model((seq_length, 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[early_stop], verbose=1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    predictions = pd.Series(predictions.flatten()).rolling(3).mean().fillna(method='bfill')
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    full_index = series.index
    test_index = full_index[seq_length + train_size:]
    train_index = full_index[:seq_length + train_size]
    plot_predictions(train_index, test_index, series[:seq_length + train_size], y_test_rescaled.flatten(), predictions, country_name)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))
    mae = mean_absolute_error(y_test_rescaled.flatten(), predictions)
    r2 = r2_score(y_test_rescaled.flatten(), predictions)
    actual_trend = np.gradient(y_test_rescaled.flatten())
    pred_trend = np.gradient(predictions)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
