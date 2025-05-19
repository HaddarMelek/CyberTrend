import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from pmdarima import auto_arima

def load_and_prepare_data_with_embeddings(file_path):
    df = pd.read_csv(file_path)
    df = df[df['country'] == 'Tunisian Republic']
    df['attackdate'] = pd.to_datetime(df['attackdate'])
    df.set_index('attackdate', inplace=True)
    df = df.sort_index()

    full_index = pd.date_range(start='2022-10-01', end='2023-12-31', freq='D')

    daily_attacks = df.groupby(df.index)['total_attacks'].sum().reindex(full_index, fill_value=0)
    daily_attacks.index.name = 'attackdate'

    df['embeddings'] = df['embeddings'].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=' ') if isinstance(x, str) else np.zeros(1)
    )
    embedding_size = len(df['embeddings'].iloc[0])

    def average_embeddings(group):
        return np.mean(np.stack(group), axis=0)

    daily_embeddings = df.groupby(df.index)['embeddings'].apply(average_embeddings).reindex(full_index)
    daily_embeddings = daily_embeddings.apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(embedding_size))

    exog_array = np.vstack(daily_embeddings.values)

    scaler = StandardScaler()
    exog_scaled = scaler.fit_transform(exog_array)

    return daily_attacks, exog_scaled

def plot_predictions(train, test, predictions, mae, r2, trend_corr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Training Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Actual Values', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test.index, y=predictions, name='Predicted Values', line=dict(color='red')))

    rmse = np.sqrt(mean_squared_error(test.values, predictions))

    fig.update_layout(
        title='Tunisian Republic - Cyber Attacks: Actual vs Predicted (ARIMA + Embeddings)',
        xaxis_title='Date',
        yaxis_title='Total Attacks',
        width=1200, height=600,
        hovermode='x unified'
    )
    fig.add_annotation(
        text=(
            f'RMSE: {rmse:.2f}<br>'
            f'MAE: {mae:.2f}<br>'
            f'R² Score: {r2:.2f}<br>'
            f'Trend Correlation: {trend_corr:.2f}'
        ),
        xref='paper', yref='paper',
        x=0.02, y=0.95, showarrow=False,
        bgcolor='white', bordercolor='black', borderwidth=1
    )

    fig.show()

def main():
    series, exog = load_and_prepare_data_with_embeddings('news_all_cleaned_processed.csv')

    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    exog_train = exog[:train_size]
    exog_test = exog[train_size:]

    print(" Tuning ARIMA parameters with auto_arima...")
    model_auto = auto_arima(train, exogenous=exog_train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    order = model_auto.order
    print(f" Best ARIMA order: {order}")

    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=order, exog=exog_train)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test), exog=exog_test)
    forecast = forecast.clip(lower=0)

    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)
    actual_trend = np.gradient(test)
    pred_trend = np.gradient(forecast)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]

    plot_predictions(train, test, forecast, mae, r2, trend_corr)

    rmse = np.sqrt(mean_squared_error(test, forecast))

    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
