import os
import torch
import pandas as pd
import json
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import MultimodalDataset
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
    # We use the full dataframe to ensure encoders cover all values
    logger.info("Initializing reference dataset...")
    ref_dataset = MultimodalDataset(
        interactions_df=df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        text_data=text_data,
        tokenizer=tokenizer
    )
    
    # 5. Load Model
    logger.info("Loading model...")
    vocab_size = len(item_id_mapper) + 1
    tabular_dim = ref_dataset.tabular_data.shape[1]
    
    # Check for encoders to set dimensions
    # Note: In dataset.py, genre_encoder is for 'track_genre', not user gender. 
    # User gender/country encoders are not explicitly handled in dataset.py shown, 
    # but train.py assumes they might exist. 
    # Let's stick to train.py logic:
    num_genders_user = len(ref_dataset.encoders['gender_encoder'].classes_) if 'gender_encoder' in ref_dataset.encoders else 1
    num_countries_user = len(ref_dataset.encoders['country_encoder'].classes_) if 'country_encoder' in ref_dataset.encoders else 1
    
    model = TwoTowerModel(
        vocab_size=vocab_size,
        num_genders=num_genders_user,
        num_countries=num_countries_user,
        tabular_input_dim=tabular_dim,
        item_embedding_dim=256,
        user_embedding_dim=256,
        use_lora=True
    )
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Model weights loaded from {args.model_path}")
    else:
        logger.warning(f"WARNING: Model checkpoint not found at {args.model_path}. Using random weights.")
        
    model.to(device)
    model.eval()
    
    return model, ref_dataset, df

def index_catalog(args, model, ref_dataset, device):
    logger.info("Indexing catalog...")
    
    # Create a dataframe of unique tracks
    unique_df = ref_dataset.interactions_df.drop_duplicates('track_id').copy()
    
    # Add dummy user info to satisfy MultimodalDataset requirements
    # We assign a unique dummy user to each row so history logic is trivial
    unique_df['user_id'] = range(len(unique_df))
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
    
    item_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding Items"):
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
            item_embeddings.append(emb.cpu())
            
    item_embeddings = torch.cat(item_embeddings, dim=0)
    
    # Save Embeddings
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    torch.save(item_embeddings, args.index_path)
    logger.info(f"Item embeddings saved to {args.index_path}")
    
    # Save Metadata (in the same order as embeddings)
    metadata = unique_df[['track_id', 'artist_name', 'track_name', 'album_name']].to_dict('records')
    meta_path = args.index_path.replace('.pt', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    logger.info(f"Metadata saved to {meta_path}")

def recommend_for_user(args, model, ref_dataset, device):
    logger.info(f"Generating recommendations for User ID: {args.user_id}")
    
    # 1. Load Item Index
    if not os.path.exists(args.index_path):
        logger.error(f"Index not found at {args.index_path}. Run with --mode index first.")
        raise FileNotFoundError(f"Index not found at {args.index_path}. Run with --mode index first.")
        
    item_embeddings = torch.load(args.index_path, map_location=device)
    
    meta_path = args.index_path.replace('.pt', '_meta.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    # 2. Get User History
    # We look up the user in the original dataframe
    user_history_df = ref_dataset.interactions_df[ref_dataset.interactions_df['user_id'] == args.user_id]
    
    if len(user_history_df) == 0:
        logger.warning(f"User {args.user_id} not found in dataset. Cannot infer history.")
        return
        
    # Sort by timestamp
    if 'timestamp' in user_history_df.columns:
        user_history_df = user_history_df.sort_values('timestamp')
        
    # Get track IDs
    history_track_ids = user_history_df['track_id'].tolist()
    print(f"\nUser History ({len(history_track_ids)} items):")
    for i, tid in enumerate(history_track_ids[-5:]): # Show last 5
        # Find metadata
        meta = next((m for m in metadata if m['track_id'] == tid), None)
        if meta:
            print(f"  {i+1}. {meta['artist_name']} - {meta['track_name']}")
        else:
            print(f"  {i+1}. {tid} (Metadata not found)")
            
    # 3. Prepare User Input
    # We need to construct the input tensors for the User Tower
    # We can use the dataset's logic manually or create a dummy row
    
    # Map track IDs to integers
    history_ints = [ref_dataset.item_id_mapper.get(tid, 0) for tid in history_track_ids]
    # Truncate/Pad
    max_len = 50 # Should match training
    if len(history_ints) > max_len:
        history_ints = history_ints[-max_len:]
    
    history_tensor = torch.tensor([history_ints], dtype=torch.long).to(device)
    
    # Get User Attributes from the dataset (processed in __init__)
    # We assume gender and country are constant for the user, so we take the first value
    # Note: ref_dataset.interactions_df already has 'gender_idx' and 'country_idx' columns
    gender_idx = user_history_df['gender_idx'].iloc[0]
    country_idx = user_history_df['country_idx'].iloc[0]
    
    user_gender = torch.tensor([gender_idx], dtype=torch.long).to(device)
    user_country = torch.tensor([country_idx], dtype=torch.long).to(device)
    
    # 4. Get User Embedding
    with torch.no_grad():
        user_emb = model.get_user_embedding(
            history_ids=history_tensor,
            user_gender=user_gender,
            user_country=user_country
        )
        
    # 5. Compute Similarity
    # (1, D) @ (N, D).T -> (1, N)
    scores = torch.matmul(user_emb, item_embeddings.to(device).t())
    scores = scores.squeeze(0) # (N,)
    
    # 6. Filter out history (optional but recommended)
    # Set score of history items to -inf
    # We need to map history track_ids to indices in the 'metadata' list
    # This is slow O(N*M), better to have a map
    track_id_to_idx = {m['track_id']: i for i, m in enumerate(metadata)}
    
    for tid in history_track_ids:
        if tid in track_id_to_idx:
            idx = track_id_to_idx[tid]
            scores[idx] = -float('inf')
            
    # 7. Top-K
    k = 10
    topk_scores, topk_indices = torch.topk(scores, k=k)
    
    print(f"\nTop {k} Recommendations:")
    for rank, idx in enumerate(topk_indices.cpu().numpy()):
        meta = metadata[idx]
        score = topk_scores[rank].item()
        print(f"  {rank+1}. {meta['artist_name']} - {meta['track_name']} (Score: {score:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Inference for Music Recommendation")
    parser.add_argument("--mode", type=str, required=True, choices=['index', 'recommend'], help="Mode: 'index' to pre-compute embeddings, 'recommend' to query")
    parser.add_argument("--data_path", type=str, default="data/spotify-kaggle/interim/lastfm_spotify_merged_0.1.csv")
    parser.add_argument("--mapper_path", type=str, default="data/spotify-kaggle/interim/item_id_mapper.json")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/")
    parser.add_argument("--lyrics_path", type=str, default="data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--index_path", type=str, default="model_cache/item_index.pt")
    parser.add_argument("--user_id", type=str, help="User ID for recommendation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load Resources
    model, ref_dataset, _ = load_resources(args, device)
    
    if args.mode == 'index':
        index_catalog(args, model, ref_dataset, device)
    elif args.mode == 'recommend':
        if not args.user_id:
            raise ValueError("User ID is required for recommendation mode")
        recommend_for_user(args, model, ref_dataset, device)

if __name__ == '__main__':
    main()
