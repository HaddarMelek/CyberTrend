import pandas as pd
import json

def merge_news_data():
    try:
        print("Reading data files...")
        cyber_df = pd.read_csv('cyber_data.csv')
        news_df = pd.read_csv('trending_news.csv')
        
        cyber_df['attackdate'] = pd.to_datetime(cyber_df['attackdate']).dt.strftime('%Y-%m-%d')
        news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')
        
        print(f"Cyber data shape: {cyber_df.shape}")
        print(f"News data shape: {news_df.shape}")
        
        merged_df = pd.merge(
            cyber_df,
            news_df[['country', 'date', 'news_titles', 'news_urls']],
            left_on=['cleaned_country', 'attackdate'],
            right_on=['country', 'date'],
            how='left'
        )
        
        merged_df = merged_df.drop(['country_y', 'date'], axis=1)
        merged_df = merged_df.rename(columns={'country_x': 'country'})
        
        empty_json_array = json.dumps([])
        merged_df['news_titles'] = merged_df['news_titles'].fillna(empty_json_array)
        merged_df['news_urls'] = merged_df['news_urls'].fillna(empty_json_array)
        
        output_file = 'cyber_news_data.csv'
        merged_df.to_csv(output_file, index=False)
        
        print(f"\nMerged data shape: {merged_df.shape}")
        print(f"Data successfully merged and saved to {output_file}")
        
        print("\nSample of merged data:")
        sample = merged_df.head(1)
        print("\nColumns:", merged_df.columns.tolist())
        print("\nFirst row news titles:", json.loads(sample['news_titles'].iloc[0]))
        
    except Exception as e:
        print(f"Error merging data: {str(e)}")

if __name__ == "__main__":
    print("Starting data merge process...")
    merge_news_data()