import pandas as pd
import feedparser
import time
import urllib.parse
import ssl
import json

ssl._create_default_https_context = ssl._create_unverified_context

def get_trending_news(country, date):
    """Get top 5 general news items for a country on a specific date."""
    try:
        query = f"{country} news {date}"
        encoded_query = urllib.parse.quote(query)
        
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        
        titles = []
        urls = []
        for entry in feed.entries[:5]:
            titles.append(entry.title)
            urls.append(entry.link)
        
        return titles, urls
    except Exception as e:
        print(f"Error getting news for {country} on {date}: {str(e)}")
        return [], []

def test_single_entry():
    """Test scraping for the first 10 entries in cyber_data.csv"""
    try:
        df = pd.read_csv('cyber_data.csv')
        test_data = []
        
        # Process first 10 rows
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            country = row['cleaned_country']
            date = row['attackdate']
            
            print(f"\nTesting news scraping for {country} on {date}")
            
            titles, urls = get_trending_news(country, date)
            
            if titles and urls:
                test_data.append({
                    'country': country,
                    'date': date,
                    'news_titles': json.dumps(titles),
                    'news_urls': json.dumps(urls)
                })
                
                print(f"\nResults for {country}:")
                for j, (title, url) in enumerate(zip(titles, urls), 1):
                    print(f"\n{j}. {title}")
                    print(f"   URL: {url}")
            
            time.sleep(1) 
        
        if test_data:
            test_df = pd.DataFrame(test_data)
            test_df.to_csv('trending_news_test.csv', index=False)
            print(f"\n✓ Test data saved to trending_news_test.csv ({len(test_data)} entries)")
        else:
            print("\n✗ No news found for test entries")
            
    except Exception as e:
        print(f"Error in test: {str(e)}")

def process_dataset():
    """Process the entire dataset and collect news articles"""
    try:
        df = pd.read_csv('cyber_data.csv')
        
        unique_combinations = df[['cleaned_country', 'attackdate']].drop_duplicates().reset_index(drop=True)
        total = len(unique_combinations)
        
        print(f"\nProcessing {total} country-date combinations...")
        
        all_data = []
        for idx, row in unique_combinations.iterrows():
            try:
                country = row['cleaned_country']
                date = row['attackdate']
                
                print(f"[{idx+1}/{total}] Processing {country} for {date}...")
                
                titles, urls = get_trending_news(country, date)
                
                if titles and urls:
                    all_data.append({
                        'country': country,
                        'date': date,
                        'news_titles': json.dumps(titles),
                        'news_urls': json.dumps(urls)
                    })
                
                time.sleep(1)  
                
            except Exception as e:
                print(f"Error processing {country}: {str(e)}")
                continue
        
        if all_data:
            news_df = pd.DataFrame(all_data)
            news_df.to_csv('trending_news.csv', index=False)
            print(f"\n✓ News data saved to trending_news.csv")
            print(f"Collected news for {len(news_df)} country-date combinations")
        else:
            print("\n✗ No news data collected!")
            
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    print("Starting news collection...")
    choice = input("Enter '1' to test first 10 entries or '2' to process entire dataset: ")
    
    if choice == '1':
        test_single_entry()
    elif choice == '2':
        process_dataset()
    else:
        print("Invalid choice. Please enter '1' or '2'")