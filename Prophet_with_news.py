import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['country'] == "Tunisian Republic"]
    df['attackdate'] = pd.to_datetime(df['attackdate'])
    df = df.sort_values('attackdate')

    full_index = pd.date_range(start='2022-10-01', end='2023-12-31', freq='D')
    daily_attacks = df.groupby('attackdate')['total_attacks'].sum().reindex(full_index, fill_value=0)
    daily_attacks = daily_attacks.reset_index()
    daily_attacks.columns = ['ds', 'y'] 
    
    embeddings = np.vstack(df['embeddings'].apply(lambda x: np.fromstring(x, sep=',') if isinstance(x, str) else np.zeros(768)).values)
    
    embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
    data_with_embeddings = pd.concat([daily_attacks, embeddings_df], axis=1)
    
    return data_with_embeddings

def train_prophet_model(train_df):
    """
    Enhanced Prophet model with better parameters for improved predictions
    """
    model = Prophet(
        changepoint_prior_scale=0.05,    
        seasonality_prior_scale=10,      
        holidays_prior_scale=10,         
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative', 
        interval_width=0.95,
        changepoint_range=0.95,          
        n_changepoints=35               
    )
    
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    for col in train_df.columns[2:]:
        model.add_regressor(
            col,
            mode='multiplicative',  
            standardize=True        
        )
    
    model.fit(train_df)
    return model

def prepare_future_dataframe(model, test_df):
    """
    Enhanced future dataframe preparation
    """
    future = model.make_future_dataframe(
        periods=len(test_df),
        freq='D',
        include_history=True
    )
    
    for col in test_df.columns[2:]:
        future[col] = pd.concat([
            pd.Series(np.zeros(len(future) - len(test_df))),
            test_df[col].reset_index(drop=True)
        ]).values
    
    return future

def plot_predictions(train, test, forecast, country_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train['ds'], 
        y=train['y'], 
        name='Training Data', 
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=test['ds'], 
        y=test['y'], 
        name='Actual Values', 
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name='Predicted Values', 
        line=dict(color='red', width=2)
    ))

    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
    mae = mean_absolute_error(test['y'], forecast['yhat'])
    r2 = r2_score(test['y'], forecast['yhat'])
    
    actual_trend = np.gradient(test['y'])
    pred_trend = np.gradient(forecast['yhat'])
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]

    fig.update_layout(
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (Prophet with Embeddings)',
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
    return rmse, mae, r2, trend_corr

def main():
    data = load_and_prepare_data('news_all_cleaned_processed.csv')

    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()

    model = train_prophet_model(train_df)

    future = prepare_future_dataframe(model, test_df)

    forecast = model.predict(future)
    
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-len(test_df):]
    forecast['y'] = test_df['y'].values

    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    rmse, mae, r2, trend_corr = plot_predictions(train_df, test_df, forecast, "Tunisian Republic")

    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
