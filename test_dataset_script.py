import pandas as pd
import torch
import numpy as np
import os
from src.dataset import MultimodalDataset

def test_dataset():
    # Crear datos dummy
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u2'],
        'track_id': ['t1', 't2', 't1', 't3', 't2'],
        'timestamp': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-01', '2021-01-03', '2021-01-02']),
        'popularity': [50, 60, 50, 70, 60],
        'danceability': [0.5, 0.6, 0.5, 0.7, 0.6],
        'energy': [0.5, 0.6, 0.5, 0.7, 0.6],
        'key': [1, 2, 1, 3, 2],
        'loudness': [-5, -4, -5, -3, -4],
        'mode': [1, 0, 1, 1, 0],
        'speechiness': [0.1, 0.2, 0.1, 0.3, 0.2],
        'acousticness': [0.1, 0.2, 0.1, 0.3, 0.2],
        'instrumentalness': [0.1, 0.2, 0.1, 0.3, 0.2],
        'liveness': [0.1, 0.2, 0.1, 0.3, 0.2],
        'valence': [0.1, 0.2, 0.1, 0.3, 0.2],
        'tempo': [120, 130, 120, 140, 130],
        'time_signature': [4, 4, 4, 4, 4],
        'duration_ms': [200000, 210000, 200000, 220000, 210000],
        'track_genre': ['pop', 'rock', 'pop', 'jazz', 'rock']
    }
    df = pd.DataFrame(data)
    
    item_map = {'t1': 1, 't2': 2, 't3': 3}
    
    # Directorios dummy
    os.makedirs('dummy_img', exist_ok=True)
    
    dataset = MultimodalDataset(
        interactions_df=df,
        item_id_mapper=item_map,
        img_dir='dummy_img',
        max_seq_len=2
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Probar __getitem__
    sample = dataset[1] # u1, t2 (segunda interacción)
    print("Sample keys:", sample.keys())
    print("User ID:", sample['user_id'])
    print("History IDs:", sample['history_ids'])
    print("Target ID:", sample['target_id'])
    print("Target Tabular shape:", sample['target_tabular'].shape)
    
    # Verificar historia
    # u1: t1 -> t2. Al pedir idx 1 (t2), historia debería ser [t1] -> [1]
    # Con padding a len 2: [1, 0]
    expected_history = torch.tensor([1, 0])
    assert torch.equal(sample['history_ids'], expected_history), f"Expected {expected_history}, got {sample['history_ids']}"
    
    print("Test passed!")

if __name__ == "__main__":
    test_dataset()
