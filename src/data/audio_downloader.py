import os
import pandas as pd
import yt_dlp
import subprocess
from pathlib import Path
import logging
import time
import random
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/audio_downloader.log"),
        logging.StreamHandler()
    ]
)

# Configuration
DATA_PATH = "data/spotify-kaggle/interim/lastfm_spotify_merged.csv"
OUTPUT_DIR = "data/audio/processed"
TEMP_DIR = "data/audio/temp"
SAMPLE_RATE = 22050
START_TIME = "00:00:30"
DURATION = "00:00:30"  # 30 seconds duration (from 00:30 to 01:00)
MAX_CONSECUTIVE_ERRORS = 10
MIN_DELAY = 2
MAX_DELAY = 6
BATCH_SIZE = 100  # Process 100 songs
BATCH_SLEEP = 60  # Sleep for 1 minute after each batch
COOKIES_PATH = "config/youtube_cookies.txt"

# List of common User-Agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
]

def ensure_dirs():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

def get_youtube_url(query):
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch1',
        'no_warnings': True,
        'user_agent': random.choice(USER_AGENTS),  # Rotate User-Agent
        'cookiefile': COOKIES_PATH,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info:
                video_info = info['entries'][0]
            else:
                video_info = info
            return video_info['webpage_url'], video_info['id']
        except Exception as e:
            logging.error(f"Error searching for {query}: {e}")
            return None, None

def process_audio(input_path, output_path):
    """
    Process audio using ffmpeg:
    1. Cut from 00:30 to 01:00
    2. Resample to 22050 Hz
    3. Convert to Mono
    """
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-i', input_path,
        '-ss', START_TIME,
        '-t', DURATION,
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',  # Mono
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False

def download_and_process_song(track_id, artist, song):
    output_filename = f"{track_id}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if os.path.exists(output_path):
        logging.info(f"Skipping {artist} - {song} (already exists)")
        return "skipped"

    query = f"{artist} - {song} Official Audio"
    logging.info(f"Processing: {query}")

    # Add a small random delay before request to be safer
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    url, video_id = get_youtube_url(query)
    if not url:
        logging.warning(f"Could not find URL for {query}")
        return False

    temp_filename = f"{track_id}_temp"
    temp_path_template = os.path.join(TEMP_DIR, temp_filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_path_template + '.%(ext)s',
        'quiet': True,
        'noplaylist': True,
        'user_agent': random.choice(USER_AGENTS),  # Rotate User-Agent
        'cookiefile': COOKIES_PATH,
    }

    downloaded_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info['ext']
            downloaded_path = f"{temp_path_template}.{ext}"

        if downloaded_path and os.path.exists(downloaded_path):
            success = process_audio(downloaded_path, output_path)
            if success:
                logging.info(f"Successfully processed {artist} - {song}")
                # Cleanup temp file
                os.remove(downloaded_path)
                return "downloaded"
            else:
                # Cleanup temp file even if processing failed
                os.remove(downloaded_path)
                return False

    except Exception as e:
        logging.error(f"Error downloading/processing {artist} - {song}: {e}")
        if downloaded_path and os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        return False
def main():
    parser = argparse.ArgumentParser(description="Download audio from YouTube based on CSV.")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing songs.")
    parser.add_argument("--end", type=int, default=None, help="End index for processing songs.")
    args = parser.parse_args()

    ensure_dirs()

    logging.info("Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {DATA_PATH}")
        return

    # Deduplicate songs based on track_id
    unique_songs = df[['track_id', 'artist_name', 'track_name']].drop_duplicates()

    # Apply range filtering
    total_songs = len(unique_songs)
    if args.end is None:
        unique_songs = unique_songs.iloc[args.start:]
        end_idx = total_songs
    else:
        unique_songs = unique_songs.iloc[args.start:args.end]
        end_idx = args.end

    logging.info(f"Processing range: {args.start} to {end_idx} (Total: {len(unique_songs)} songs to process)")

    consecutive_errors = 0
    consecutive_errors = 0
    songs_processed_in_batch = 0

    for index, row in unique_songs.iterrows():
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logging.critical(f"Stopping execution after {MAX_CONSECUTIVE_ERRORS} consecutive errors. Likely rate limited or banned.")
            break

        # Batch processing logic
        if songs_processed_in_batch >= BATCH_SIZE:
            logging.info(f"Batch of {BATCH_SIZE} songs reached. Sleeping for {BATCH_SLEEP} seconds to avoid rate limiting...")
            time.sleep(BATCH_SLEEP)
            songs_processed_in_batch = 0

        result = download_and_process_song(row['track_id'], row['artist_name'], row['track_name'])

        if result == "downloaded":
            consecutive_errors = 0
            songs_processed_in_batch += 1
        elif result == "skipped":
            consecutive_errors = 0
            # Do not increment batch count for skipped songs
        else:
            consecutive_errors += 1
            logging.warning(f"Consecutive error count: {consecutive_errors}")


if __name__ == "__main__":
    main()
