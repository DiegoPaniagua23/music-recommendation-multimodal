import pandas as pd
import os

DATA_PATH = "data/spotify-kaggle/interim/lastfm_spotify_merged.csv"
OUTPUT_DIR = "data/audio/processed"

def check_missing():
    df = pd.read_csv(DATA_PATH)
    unique_songs = df[['track_id', 'artist_name', 'track_name']].drop_duplicates().reset_index(drop=True)

    # Check first 5500
    limit = 5500
    missing_count = 0

    print(f"Checking first {limit} songs...")

    expected_files = set()
    for i in range(limit):
        row = unique_songs.iloc[i]
        expected_files.add(row['track_id'])

    print(f"Unique track_ids in first {limit} rows: {len(expected_files)}")

    missing_count = 0
    for track_id in expected_files:
        path = os.path.join(OUTPUT_DIR, f"{track_id}.mp3")
        if not os.path.exists(path):
            missing_count += 1

    print(f"Missing files for first {limit} rows: {missing_count}")
    print(f"Files found: {len(expected_files) - missing_count}")

if __name__ == "__main__":
    check_missing()
