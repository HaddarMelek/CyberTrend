import pandas as pd
import numpy as np
from pmdarima import auto_arima
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def plot_predictions(train, test, predictions, country_name):
    """
    Enhanced visualization with trend analysis and dynamic country name
    """
    fig = go.Figure()
    
    # Training data (blue)
    fig.add_trace(go.Scatter(
        x=train.index, 
        y=train.values, 
        name='Training Data', 
        line=dict(color='blue', width=1)
    ))
    
    # Actual values (solid green)
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=test.values, 
        name='Actual Values', 
        line=dict(color='green', width=2)
    ))
    
    # Predicted values (red)
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=predictions, 
        name='Predicted Values', 
        line=dict(color='red', width=2)
    ))

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test.values, predictions))
    mae = mean_absolute_error(test.values, predictions)
    r2 = r2_score(test.values, predictions)
    
    # Calculate trend correlation
    actual_trend = np.gradient(test.values)
    pred_trend = np.gradient(predictions)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]

    fig.update_layout(
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (ARIMA)',
        xaxis_title='Date',
        yaxis_title='Total Attacks',
        width=1200,
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add metrics annotation with updated position
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
    # Get country name from the data
    country_name = "Tunisian Republic"
    series = load_and_prepare_data('news_all_cleaned_processed.csv')
    
    # Split data
    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    
    # Fit model and make predictions
    model = auto_arima(train, seasonal=False, suppress_warnings=True, stepwise=True)
    forecast = model.predict(n_periods=len(test))
    forecast = np.clip(forecast, 0, None)
    
    # Plot with updated metrics and display
    plot_predictions(train, test, forecast, country_name)
    
    # Calculate and print metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)
    actual_trend = np.gradient(test)
    pred_trend = np.gradient(forecast)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]
    
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
