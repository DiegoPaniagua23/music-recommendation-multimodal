import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dataset import MultimodalDataset

def test_dataset():
    print("=== Testing MultimodalDataset Loading ===")

    # Paths
    DATA_DIR = Path("data")
    INTERIM_DATA_PATH = DATA_DIR / "spotify-kaggle/interim/lastfm_spotify_merged.csv"
    IMG_DIR = DATA_DIR / "spotify-kaggle/album_covers"
    AUDIO_DIR = DATA_DIR / "audio/mels"

    # Load Interactions
    print(f"Loading interactions from {INTERIM_DATA_PATH}...")
    df = pd.read_csv(INTERIM_DATA_PATH)

    # Create simple mapper
    unique_tracks = df['track_id'].unique()
    item_id_mapper = {tid: i+1 for i, tid in enumerate(unique_tracks)} # 1-based index

    print(f"Total unique tracks: {len(unique_tracks)}")

    # Initialize Dataset
    print("Initializing MultimodalDataset...")
    dataset = MultimodalDataset(
        interactions_df=df, # Use full dataframe
        item_id_mapper=item_id_mapper,
        img_dir=str(IMG_DIR),
        audio_dir=str(AUDIO_DIR),
        max_seq_len=10
    )

    print(f"Dataset length: {len(dataset)}")

    # 1. Test Single Item Structure
    print("\n--- Checking Single Item Structure ---")
    idx = 0
    item = dataset[idx]
    print(f"User ID: {item['user_id']}")
    print(f"Target ID (Mapped): {item['target_id']}")
    print(f"Audio Shape: {item['target_audio'].shape}")
    print(f"Image Shape: {item['target_image'].shape}")

    expected_audio_shape = (1, 128, 128)
    if item['target_audio'].shape == expected_audio_shape:
        print("✅ Audio dimension correct.")
    else:
        print(f"❌ Audio dimension INCORRECT. Expected: {expected_audio_shape}, Got: {item['target_audio'].shape}")

    # 2. Test DataLoader Batching
    print("\n--- Checking DataLoader (Batch=4) ---")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch Audio Shape: {batch['target_audio'].shape}")

    if batch['target_audio'].shape == (4, 1, 128, 128):
        print("✅ Batch shape correct.")
    else:
        print("❌ Batch shape incorrect.")

    # 3. Check Audio Coverage (Random Sample)
    print("\n--- Checking Audio Availability (Sample of 100 items) ---")
    found_audio = 0
    sample_size = 100

    # Random indices
    indices = torch.randperm(len(dataset))[:sample_size]

    for i in tqdm(indices, desc="Checking audio existence"):
        item = dataset[i.item()]
        # Check if not all zeros
        if torch.sum(torch.abs(item['target_audio'])) > 0:
            found_audio += 1

    print(f"\nFound audio for {found_audio} out of {sample_size} sampled items.")
    print(f"Estimated Coverage: {(found_audio/sample_size)*100:.1f}%")

if __name__ == "__main__":
    test_dataset()
