import lightgbm as lgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_prepare_data_with_embeddings(file_path):
    df = pd.read_csv(file_path)
    df = df[df['country'] == "Tunisian Republic"]
    df['attackdate'] = pd.to_datetime(df['attackdate'])
    df.set_index('attackdate', inplace=True)
    df = df.sort_index()
    full_index = pd.date_range(start='2022-10-01', end='2023-12-31', freq='D')
    daily_attacks = df.groupby(df.index)['total_attacks'].sum()
    daily_attacks = daily_attacks.reindex(full_index, fill_value=0)
    daily_attacks.index.name = 'attackdate'
    df['embeddings'] = df['embeddings'].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=' ') if isinstance(x, str) else np.zeros(1)
    )
    return daily_attacks, df

def create_lag_features(series, lags=[1, 2, 3, 5, 7]):
    """
    Simplified and focused feature engineering
    """
    df = pd.DataFrame({'total_attacks': series})
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    for lag in lags:
        df[f'lag_{lag}'] = df['total_attacks'].shift(lag)
    
    df['ma_3d'] = df['total_attacks'].rolling(window=3).mean()
    df['ma_7d'] = df['total_attacks'].rolling(window=7).mean()
    
    df['trend'] = df['total_attacks'].diff()
    
    df.dropna(inplace=True)
    return df

def train_lightgbm_model(X_train, y_train):
    """
    Refined LightGBM parameters for better prediction accuracy
    """
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,
        'min_gain_to_split': 0.0001,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'n_estimators': 500,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_train, y_train)],
        eval_metric='rmse'
    )
    
    return model

def plot_predictions(train, test, predictions, country_name):
    """
    Enhanced visualization with trend analysis and dynamic country name
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
        title=f'{country_name} - Cyber Attacks: Actual vs Predicted (LightGBM with Embeddings)',
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
        align='left',
        xanchor='left'
    )
    
    fig.show()

def prepare_embeddings_data(embeddings_scaled, common_index):
    """
    Prepare embeddings dataframe with proper index alignment
    """
    embeddings_df = pd.DataFrame(embeddings_scaled)
    print(f"Embeddings shape: {embeddings_scaled.shape}")
    print(f"Index length: {len(common_index)}")
    
    if len(embeddings_df) > len(common_index):
        print("Truncating embeddings to match index length")
        embeddings_df = embeddings_df.iloc[:len(common_index)]
    elif len(embeddings_df) < len(common_index):
        print("Warning: Embeddings shorter than index")
        return None
    
    embeddings_df.index = common_index
    
    return embeddings_df

def main():
    country_name = "Tunisian Republic" 
    series, df = load_and_prepare_data_with_embeddings('news_all_cleaned_processed.csv')
    
    df_lags = create_lag_features(series)
    
    embedding_size = len(df['embeddings'].iloc[0])
    common_index = df_lags.index.intersection(df.index)
    df_lags_filtered = df_lags.loc[common_index]
    
    embeddings_series = df.loc[common_index, 'embeddings'].apply(
        lambda x: x if isinstance(x, np.ndarray) and x.shape[0] == embedding_size else np.zeros(embedding_size)
    )
    embeddings = np.vstack(embeddings_series.values)
    
    scaler = StandardScaler()
    numeric_cols = df_lags.select_dtypes(include=[np.number]).columns
    df_lags[numeric_cols] = scaler.fit_transform(df_lags[numeric_cols])
    
    pca = PCA(n_components=50)  
    embeddings_scaled = scaler.fit_transform(embeddings)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    embeddings_reduced_df = pd.DataFrame(
        embeddings_reduced, 
        index=common_index,
        columns=[f'emb_pca_{i}' for i in range(50)]
    )
    
    df_final = pd.concat([df_lags_filtered, embeddings_reduced_df], axis=1)
    
    train_size = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:train_size].copy()
    test_df = df_final.iloc[train_size:].copy()
    
    scaler = StandardScaler()
    X_train = train_df.drop(columns='total_attacks')
    X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                          columns=X_train.columns, 
                          index=X_train.index)
    
    X_test = test_df.drop(columns='total_attacks')
    X_test = pd.DataFrame(scaler.transform(X_test), 
                         columns=X_test.columns, 
                         index=X_test.index)
    
    y_train = train_df['total_attacks']
    y_test = test_df['total_attacks']
    
    model = train_lightgbm_model(X_train, y_train)
    predictions = model.predict(X_test)
    
    predictions = np.maximum(predictions, 0)
    predictions_series = pd.Series(predictions, index=X_test.index)
    
    plot_predictions(train_df, test_df, predictions_series, country_name)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nTest Set Evaluation:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
