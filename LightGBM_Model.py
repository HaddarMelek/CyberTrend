import lightgbm as lgb
import pandas as pd
import numpy as np
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

def create_lag_features(series, lags=[1, 2, 3]):
    df = pd.DataFrame({'total_attacks': series})
    for lag in lags:
        df[f'lag_{lag}'] = df['total_attacks'].shift(lag)
    df.dropna(inplace=True)
    return df

def train_lightgbm_model(X_train, y_train):
    model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', metric='rmse')
    model.fit(X_train, y_train)
    return model

def plot_predictions(train, test, predictions, country_name):
    """
    Plot predictions with updated metrics and dynamic country name
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train.index, 
        y=train['total_attacks'], 
        name='Training Data', 
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=test['total_attacks'], 
        name='Actual Values', 
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=predictions, 
        name='Predicted Values', 
        line=dict(color='red', width=2)
    ))

    rmse = np.sqrt(mean_squared_error(test['total_attacks'], predictions))
    mae = mean_absolute_error(test['total_attacks'], predictions)
    r2 = r2_score(test['total_attacks'], predictions)
    
    actual_trend = np.gradient(test['total_attacks'])
    pred_trend = np.gradient(predictions)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]

    fig.update_layout(
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (LightGBM without Embeddings)',
        xaxis_title='Date',
        yaxis_title='Total Attacks',
        width=1200, 
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",  
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
        align='left',    
        xanchor='left'   
    )
    
    fig.show()

def main():
    country_name = "Tunisian Republic"  
    series = load_and_prepare_data('news_all_cleaned_processed.csv')
    df = create_lag_features(series)

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df.drop(columns='total_attacks')
    y_train = train_df['total_attacks']
    X_test = test_df.drop(columns='total_attacks')
    y_test = test_df['total_attacks']

    model = train_lightgbm_model(X_train, y_train)
    predictions = model.predict(X_test)

    predictions_series = pd.Series(predictions, index=X_test.index)

    plot_predictions(train_df, test_df, predictions_series, country_name)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    actual_trend = np.gradient(y_test)
    pred_trend = np.gradient(predictions)
    trend_corr = np.corrcoef(actual_trend, pred_trend)[0,1]
    
    print(f"\nEvaluation Results:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Trend Correlation: {trend_corr:.4f}")

if __name__ == "__main__":
    main()
