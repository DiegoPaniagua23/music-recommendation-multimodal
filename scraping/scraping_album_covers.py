import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

IMAGE_FOLDER='./data/spotify-kaggle/album_covers/'
FAILED_IDS_FILE='./data/spotify-kaggle/interim/failed_ids.txt'

def get_spotify_client():
    """Crea y devuelve un cliente de Spotify usando las credenciales del cliente."""
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                                           client_secret=SPOTIFY_CLIENT_SECRET))
    return sp

def load_bad_ids():
    if os.path.exists(FAILED_IDS_FILE):
        with open(FAILED_IDS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_bad_ids(bad_ids):
    with open(FAILED_IDS_FILE, 'w') as f:
        for _id in bad_ids:
            f.write(f"{_id}\n")

def process_batch(sp : spotipy.Spotify, batch_ids, bad_ids, valid_ids):
    ids_to_fetch = []
    for _id in batch_ids:
        if os.path.exists(f'{IMAGE_FOLDER}/{_id}.jpg'):
            valid_ids.add(_id)
            continue
        if _id in bad_ids:
            continue
        ids_to_fetch.append(_id)
    
    if not ids_to_fetch:
        return

    try:
        response = sp.tracks(ids_to_fetch)
        for _id, track in zip(ids_to_fetch, response['tracks']):
            if track is None:
                # print(f"Track {_id} not found.")
                bad_ids.add(_id)
                continue
            
            try:
                url_found = next((img['url'] for img in track['album']['images'] if img['height'] == 300), None)
                if url_found:
                    img_data = requests.get(url_found, timeout=10).content
                    with open(f'{IMAGE_FOLDER}/{_id}.jpg', 'wb') as handler:
                        handler.write(img_data)
                    valid_ids.add(_id)
                else:
                    # print(f"No 300px image found for track {_id}.")
                    bad_ids.add(_id)
            except Exception as e:
                print(f"Error procesando la cancion {_id}: {e}")
    except Exception as e:
        print(f"Batch request failed: {e}")

def clean_image_folder(image_folder: str, valid_ids: set):
    """Eliminar imágenes en la carpeta que no están en el conjunto valid_ids."""
    if not os.path.exists(image_folder):
        return
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            track_id = filename[:-4]  # Remove .jpg extension
            if track_id not in valid_ids:
                os.remove(os.path.join(image_folder, filename))
                print(f"Se eliminó por no ser válido: {filename}")

def main():
    sp = get_spotify_client()
    
    # Load track IDs from CSV
    df_tracks = pd.read_csv('./data/spotify-kaggle/interim/songs_catalog.csv')
    track_ids = df_tracks['track_id'].tolist()
    
    bad_ids = load_bad_ids()
    valid_ids = set()
    
    BATCH_SIZE = 50
    chunks = [track_ids[i:i + BATCH_SIZE] for i in range(0, len(track_ids), BATCH_SIZE)]
    
    print(f"Processing {len(track_ids)} tracks in {len(chunks)} batches...")
    for chunk in tqdm(chunks):
        process_batch(sp, chunk, bad_ids, valid_ids)
    
    save_bad_ids(bad_ids)
    clean_image_folder(IMAGE_FOLDER, valid_ids)
    
    print(f"Procesamiento completado. Imágenes válidas: {len(valid_ids)}, IDs fallidos: {len(bad_ids)}")

if __name__ == "__main__":
    main()