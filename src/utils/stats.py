import pandas as pd

DATA_PATH = "data/spotify-kaggle/interim/lastfm_spotify_merged.csv"

try:
    df = pd.read_csv(DATA_PATH)
    unique_songs = df[['track_id', 'artist_name', 'track_name']].drop_duplicates()
    print(f"Total unique songs: {len(unique_songs)}")
except Exception as e:
    print(f"Error reading dataset: {e}")
