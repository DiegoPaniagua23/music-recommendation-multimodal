import os
import argparse
import pandas as pd
import json
import joblib
import logging
from src.dataset import MultimodalDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate and save encoders for a dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--mapper_path", type=str, required=True, help="Path to the item_id_mapper.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save encoders.pkl")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/", help="Path to images (dummy for init)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    logger.info(f"Loading mapper from {args.mapper_path}...")
    with open(args.mapper_path, 'r') as f:
        item_id_mapper = json.load(f)
        
    # --- REPLICATE TRAIN.PY SPLIT LOGIC ---
    # Split Train/Val (Por tiempo)
    # Ordenamos por timestamp para respetar la secuencia temporal
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    split_idx = int(len(df) * 0.8) # 80% Train, 20% Val
    train_df = df.iloc[:split_idx].copy()
    
    logger.info(f"Using first 80% of data ({len(train_df)} rows) to fit encoders, matching training logic.")
    # --------------------------------------

    # Initialize dataset to fit encoders
    # We don't need text_data or tokenizer for fitting tabular encoders
    logger.info("Initializing dataset to fit encoders...")
    dataset = MultimodalDataset(
        interactions_df=train_df, # Use train_df instead of full df
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        text_data={}, # Dummy
        tokenizer=None # Dummy
    )
    
    encoders_path = os.path.join(args.output_dir, 'encoders.pkl')
    logger.info(f"Saving encoders to {encoders_path}...")
    joblib.dump(dataset.encoders, encoders_path)
    logger.info("Done!")

if __name__ == "__main__":
    main()
