import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# IGDB API credentials
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

# IGDB API endpoints
AUTH_URL = 'https://id.twitch.tv/oauth2/token'
API_ENDPOINT = 'https://api.igdb.com/v4/games'

def get_access_token():
    auth_params = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'client_credentials'
    }
    auth_response = requests.post(AUTH_URL, params=auth_params)
    return auth_response.json()['access_token']

def fetch_games(access_token, time_back: int=365, offset=0):
    headers = {
        'Client-ID': CLIENT_ID,
        'Authorization': f'Bearer {access_token}',
    }
    
    # Calculate the timestamp for 10 years ago
    limit_time = int((datetime.now() - timedelta(days=time_back)).timestamp())
    current_time = int(datetime.now().timestamp())
    
    body = f"""
    fields name,first_release_date,release_dates.date,genres.name,platforms.name,summary,cover.url;
    where first_release_date >= {limit_time} & release_dates.date <= {current_time};
    sort first_release_dates.date desc;
    limit 500;
    offset {offset};
    """
    
    response = requests.post(API_ENDPOINT, headers=headers, data=body)
    return response.json()

def get_games(time_back: int):
    access_token = get_access_token()
    all_games = []
    offset = 0
    
    while True:
        games = fetch_games(access_token, time_back, offset)
        if not games:
            break
        
        all_games.extend(games)
        print(f"Fetched {len(games)} games. Total: {len(all_games)}")
        
        if len(games) < 500:
            break
        
        offset += 500
        time.sleep(0.3)  # Respect rate limits
    
    # Save all games to a JSON file
    print(f"Scraped a total of {len(all_games)} games from the last {time_back} days.")
    return all_games

if __name__ == "__main__":
    games = get_games(5)
    df = pd.read_json('games_entry.json')
    pd.get_dummies(df['genres'])
    df.to_parquet('df.parquet')