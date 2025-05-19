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
    daily_attacks.index.name = 'attackdate'
    return daily_attacks

def create_sequences_with_embeddings(data, embeddings, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        if i + seq_length >= len(embeddings):
            break
        X_data = data[i:i+seq_length]
        X_embed = embeddings[i:i+seq_length]
        X.append(np.concatenate((X_data, X_embed), axis=1))
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
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
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (LSTM with Embeddings)',
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
    df = pd.read_csv('news_all_cleaned_processed.csv')
    df = df[df['country'] == country_name]
    df['attackdate'] = pd.to_datetime(df['attackdate'])
    df = df.sort_values('attackdate')
    embeddings = np.vstack(df['embeddings'].apply(lambda x: np.fromstring(x, sep=',') if isinstance(x, str) else np.zeros(768)).values)
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    embedding_start_date = df['attackdate'].min().normalize()
    embedding_end_date = df['attackdate'].max().normalize()
    series = series[embedding_start_date:embedding_end_date]
    scaled_values = scaled_values[-len(series):]
    X, y = create_sequences_with_embeddings(scaled_values, embeddings, seq_length=7)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[early_stop])
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    full_index = series.index
    train_index = full_index[:train_size + 7]
    test_index = full_index[train_size + 7:]
    plot_predictions(train_index, test_index, series[:train_size + 7], y_test_rescaled.flatten(), predictions.flatten(), country_name)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))
    mae = mean_absolute_error(y_test_rescaled.flatten(), predictions.flatten())
    r2 = r2_score(y_test_rescaled.flatten(), predictions.flatten())
    actual_trend = np.gradient(y_test_rescaled.flatten())
    pred_trend = np.gradient(predictions.flatten())
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
