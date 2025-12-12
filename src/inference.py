import os
import torch
import pandas as pd
import json
import joblib
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch.nn.functional as F

from src.data.dataset import MultimodalDataset
from src.models import TwoTowerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_resources(args, device):
    logger.info("Loading resources...")

    # 1. Load Data (needed for encoders and mapper)
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found at {args.data_path}")
        raise FileNotFoundError(f"Data file not found at {args.data_path}")

    df = pd.read_csv(args.data_path)

    # 2. Load Mapper
    if not os.path.exists(args.mapper_path):
        logger.error(f"Mapper not found at {args.mapper_path}. Run training first.")
        raise FileNotFoundError(f"Mapper not found at {args.mapper_path}. Run training first.")

    with open(args.mapper_path, 'r') as f:
        item_id_mapper = json.load(f)

    # 3. Prepare Text Data (Same logic as train.py)
    logger.info("Preparing text data...")
    cache_dir = os.getenv('HF_HOME')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False, cache_dir=cache_dir)

    unique_tracks = df.drop_duplicates('track_id')
    texts = (unique_tracks['artist_name'].fillna("") + " - " +
             unique_tracks['track_name'].fillna("") + " (" +
             unique_tracks['album_name'].fillna("") + ")")
    text_data = dict(zip(unique_tracks['track_id'], texts))

    # Load lyrics if available
    if args.lyrics_path and os.path.exists(args.lyrics_path):
        logger.info(f"Loading lyrics from {args.lyrics_path}...")
        lyrics_df = pd.read_csv(args.lyrics_path)
        if 'track_id' in lyrics_df.columns and 'lyrics' in lyrics_df.columns:
            lyrics_df = lyrics_df.dropna(subset=['lyrics'])
            lyrics_dict = dict(zip(lyrics_df['track_id'], lyrics_df['lyrics']))
            text_data.update(lyrics_dict)

    # 4. Initialize Reference Dataset (to fit encoders)
    # Load encoders if available to ensure consistency with training
    if args.encoders_path:
        encoders_path = args.encoders_path
    else:
        encoders_path = os.path.join(os.path.dirname(args.mapper_path), 'encoders.pkl')

    encoders = None
    if os.path.exists(encoders_path):
        logger.info(f"Loading encoders from {encoders_path}...")
        encoders = joblib.load(encoders_path)
    else:
        logger.warning(f"Encoders not found at {encoders_path}. Encoders will be refitted (may cause dimension mismatch).")

    logger.info("Initializing reference dataset...")
    ref_dataset = MultimodalDataset(
        interactions_df=df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        text_data=text_data,
        tokenizer=tokenizer,
        encoders=encoders
    )

    # 5. Load Model
    logger.info("Loading model...")

    # Load state dict first to inspect dimensions
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found at {args.model_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=device)

    # Infer dimensions from state_dict to ensure compatibility
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # User Tower: country_embedding.weight -> (num_countries, embedding_dim)
    num_countries_ckpt = state_dict['user_tower.country_embedding.weight'].shape[0]

    # Item Tower: tabular_encoder.mlp.0.weight -> (hidden_dim, input_dim)
    tabular_dim_ckpt = state_dict['item_tower.tabular_encoder.mlp.0.weight'].shape[1]

    vocab_size = len(item_id_mapper) + 1

    # Check for encoders to set dimensions
    num_genders_user = len(ref_dataset.encoders['gender_encoder'].classes_) if 'gender_encoder' in ref_dataset.encoders else 1

    num_countries_user = num_countries_ckpt
    tabular_dim = tabular_dim_ckpt

    logger.info(f"Model Dimensions - Vocab: {vocab_size}, Tabular: {tabular_dim}, Countries: {num_countries_user}")

    model = TwoTowerModel(
        vocab_size=vocab_size,
        num_genders=num_genders_user,
        num_countries=num_countries_user,
        tabular_input_dim=tabular_dim,
        item_embedding_dim=256,
        user_embedding_dim=256,
        use_lora=True
    )

    model.load_state_dict(state_dict)
    logger.info(f"Model weights loaded from {args.model_path}")

    model.to(device)
    model.eval()

    return model, ref_dataset, df

def index_catalog(args, model, ref_dataset, original_df, device):
    logger.info("Indexing catalog (Dense Matrix Strategy)...")

    # Create a dataframe of unique tracks from the ORIGINAL dataframe
    unique_df = original_df.drop_duplicates('track_id').copy()

    # Add dummy user info to satisfy MultimodalDataset requirements
    unique_df['user_id'] = 'dummy_user'
    unique_df['timestamp'] = '2020-01-01'

    # Create dataset for indexing
    catalog_dataset = MultimodalDataset(
        interactions_df=unique_df,
        item_id_mapper=ref_dataset.item_id_mapper,
        img_dir=ref_dataset.img_dir,
        text_data=ref_dataset.text_data,
        tokenizer=ref_dataset.tokenizer,
        encoders=ref_dataset.encoders # Use fitted encoders
    )

    loader = DataLoader(catalog_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    embeddings_list = []
    track_ids_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding Items"):
            # Save mapped integer IDs to place them correctly later
            track_ids_list.append(batch['target_id'])

            # Move item features to device
            images = batch['target_image'].to(device)
            audio = batch['target_audio'].to(device) if batch['target_audio'] is not None else None
            input_ids = batch['target_input_ids'].to(device)
            attention_mask = batch['target_attention_mask'].to(device)
            tabular = batch['target_tabular'].to(device)

            emb = model.get_item_embedding(
                images=images,
                audio=audio,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tabular=tabular
            )

            # Handle NaNs
            if torch.isnan(emb).any():
                emb = torch.nan_to_num(emb, nan=0.0)

            # Normalize embeddings (Cosine Similarity)
            emb = F.normalize(emb, p=2, dim=1, eps=1e-8)

            embeddings_list.append(emb.cpu())

    # Concatenate all batches
    all_embeddings = torch.cat(embeddings_list, dim=0) # (M, D)
    all_ids = torch.cat(track_ids_list, dim=0)         # (M,)

    # Create Dense Matrix (Vocab_Size, D)
    # Index 0 is padding/unknown
    vocab_size = len(ref_dataset.item_id_mapper) + 1
    embedding_dim = all_embeddings.shape[1]

    dense_embeddings = torch.zeros(vocab_size, embedding_dim)

    # Scatter embeddings to their correct integer ID positions
    # This ensures dense_embeddings[i] corresponds to item_id_mapper ID 'i'
    dense_embeddings[all_ids.long()] = all_embeddings

    # Save Embeddings
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    torch.save(dense_embeddings, args.index_path)
    logger.info(f"Dense item index saved to {args.index_path} (Shape: {dense_embeddings.shape})")

    # We don't need _meta.json anymore because we can map Int -> TrackID -> DF Metadata

def recommend_for_user(args, model, ref_dataset, device):
    logger.info(f"Generating recommendations for User ID: {args.user_id}")

    # 1. Load Item Index
    if not os.path.exists(args.index_path):
        logger.error(f"Index not found at {args.index_path}. Run with --mode index first.")
        raise FileNotFoundError(f"Index not found at {args.index_path}. Run with --mode index first.")

    # Load dense tensor (Vocab_Size, D)
    item_embeddings = torch.load(args.index_path, map_location=device)

    # 2. Prepare Metadata Lookup
    # Create a fast lookup dict: track_id -> {artist, name, album}
    # We use the dataframe loaded in load_resources
    logger.info("Building metadata lookup...")
    meta_df = ref_dataset.interactions_df[['track_id', 'artist_name', 'track_name', 'album_name']].drop_duplicates('track_id')
    meta_dict = meta_df.set_index('track_id').to_dict('index')

    # Invert mapper: Int ID -> Track ID string
    id_to_track = {v: k for k, v in ref_dataset.item_id_mapper.items()}

    # 3. Get User History
    user_history_df = ref_dataset.interactions_df[ref_dataset.interactions_df['user_id'] == args.user_id]

    if len(user_history_df) == 0:
        logger.warning(f"User {args.user_id} not found in dataset. Cannot infer history.")
        return

    # Sort by timestamp
    if 'timestamp' in user_history_df.columns:
        user_history_df = user_history_df.sort_values('timestamp')

    # Get track IDs
    history_track_ids = user_history_df['track_id'].tolist()
    print(f"\nTotal User History ({len(history_track_ids)} items)")

    # 4. Prepare User Input
    # Map track IDs to integers
    history_ints = [ref_dataset.item_id_mapper.get(tid, 0) for tid in history_track_ids]

    # Truncate to max_len (same as training)
    max_len = 50

    used_history_track_ids = history_track_ids
    if len(history_ints) > max_len:
        history_ints = history_ints[-max_len:]
        used_history_track_ids = history_track_ids[-max_len:]

    print(f"\nUser History Considered (Last {len(used_history_track_ids)} items):")
    for i, tid in enumerate(used_history_track_ids):
        meta = meta_dict.get(tid)
        if meta:
            print(f"  {i+1}. {meta['artist_name']} - {meta['track_name']}")
        else:
            print(f"  {i+1}. {tid} (Metadata not found)")

    history_tensor = torch.tensor([history_ints], dtype=torch.long).to(device)

    # Get User Attributes
    gender_idx = user_history_df['gender_idx'].iloc[0]
    country_idx = user_history_df['country_idx'].iloc[0]

    user_gender = torch.tensor([gender_idx], dtype=torch.long).to(device)
    user_country = torch.tensor([country_idx], dtype=torch.long).to(device)

    # 5. Get User Embedding
    with torch.no_grad():
        user_emb = model.get_user_embedding(
            history_ids=history_tensor,
            user_gender=user_gender,
            user_country=user_country
        )
        # Normalize user embedding
        user_emb = F.normalize(user_emb, p=2, dim=1, eps=1e-8)

    # 6. Compute Similarity
    # (1, D) @ (Vocab, D).T -> (1, Vocab)
    # The indices of 'scores' correspond exactly to the Integer IDs in the mapper
    scores = torch.matmul(user_emb, item_embeddings.t())
    scores = scores.squeeze(0) # (Vocab,)

    # 7. Filter out history and padding
    # Mask padding (index 0)
    scores[0] = -float('inf')

    # Mask history items
    # We already have history_ints
    for h_idx in history_ints:
        if h_idx < len(scores):
            scores[h_idx] = -float('inf')

    # 8. Top-K
    k = 10
    topk_scores, topk_indices = torch.topk(scores, k=k)

    print(f"\nTop {k} Recommendations:")
    for rank, idx_tensor in enumerate(topk_indices):
        idx = idx_tensor.item()
        score = topk_scores[rank].item()

        # Convert Int ID -> Track ID
        track_id = id_to_track.get(idx)

        if track_id:
            meta = meta_dict.get(track_id)
            if meta:
                print(f"  {rank+1}. {meta['artist_name']} - {meta['track_name']} (Score: {score:.4f})")
            else:
                print(f"  {rank+1}. Track ID: {track_id} (Metadata not found) (Score: {score:.4f})")
        else:
            print(f"  {rank+1}. Unknown Index: {idx} (Score: {score:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Inference for Music Recommendation")
    parser.add_argument("--mode", type=str, required=True, choices=['index', 'recommend'], help="Mode: 'index' to pre-compute embeddings, 'recommend' to query")
    parser.add_argument("--data_path", type=str, default="data/spotify-kaggle/interim/lastfm_spotify_merged.csv")
    parser.add_argument("--mapper_path", type=str, default="data/spotify-kaggle/interim/item_id_mapper.json")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/")
    parser.add_argument("--lyrics_path", type=str, default="data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/prueba_10porciento/best_model_0.1_test.pth")
    parser.add_argument("--encoders_path", type=str, default=None, help="Path to saved encoders.pkl")
    parser.add_argument("--index_path", type=str, default="checkpoints/prueba_10porciento/item_index.pt")
    parser.add_argument("--user_id", type=str, help="User ID for recommendation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load Resources
    model, ref_dataset, original_df = load_resources(args, device)

    if args.mode == 'index':
        index_catalog(args, model, ref_dataset, original_df, device)
    elif args.mode == 'recommend':
        if not args.user_id:
            raise ValueError("User ID is required for recommendation mode")
        recommend_for_user(args, model, ref_dataset, device)

if __name__ == '__main__':
    main()
