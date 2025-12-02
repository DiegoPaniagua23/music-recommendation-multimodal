import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import json
import argparse
from transformers import AutoTokenizer

from src.dataset import MultimodalDataset
from src.models import TwoTowerModel

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Mover batch a dispositivo
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss, logits, _, _ = model(batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, k=10):
    model.eval()
    # Evaluación simplificada: Recall@K "in-batch"
    # Verificamos si el item positivo está en el top-K de items del batch para ese usuario.
    
    hits = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k_key, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k_key] = v.to(device)
            
            _, logits, _, _ = model(batch)
            
            # logits: (B, B)
            # Diagonal contiene los pares positivos (usuario i, item i)
            
            # Obtener top-k índices para cada usuario (fila)
            # topk_indices: (B, K)
            _, topk_indices = torch.topk(logits, k=k, dim=1)
            
            # Targets son 0, 1, 2, ... B-1 (la diagonal)
            targets = torch.arange(logits.shape[0], device=device).unsqueeze(1) # (B, 1)
            
            # Verificar si el target está en topk
            is_hit = (topk_indices == targets).any(dim=1)
            
            hits += is_hit.sum().item()
            total += logits.shape[0]
            
    return hits / total if total > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Multimodal Model")
    parser.add_argument("--data_path", type=str, default="data/spotify-kaggle/interim/lastfm_spotify_merged.csv")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/")
    parser.add_argument("--audio_dir", type=str, default=None, help="Path to audio embeddings dir")
    parser.add_argument("--text_embeddings_path", type=str, default=None, help="Path to pre-computed text embeddings .pt file")
    parser.add_argument("--mapper_path", type=str, default="data/spotify-kaggle/interim/item_id_mapper.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Cargar Datos
    print("Cargando datos...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")
        
    df = pd.read_csv(args.data_path)
    
    # Cargar o Crear Mapper
    if os.path.exists(args.mapper_path):
        print(f"Cargando mapper desde {args.mapper_path}")
        with open(args.mapper_path, 'r') as f:
            item_id_mapper = json.load(f)
    else:
        print("Mapper no encontrado, creando desde datos...")
        unique_tracks = df['track_id'].unique()
        item_id_mapper = {tid: i+1 for i, tid in enumerate(unique_tracks)}
        # Guardar mapper
        os.makedirs(os.path.dirname(args.mapper_path), exist_ok=True)
        with open(args.mapper_path, 'w') as f:
            json.dump(item_id_mapper, f)
            
    # Split Train/Val (Por tiempo)
    # Ordenamos por timestamp para respetar la secuencia temporal
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    split_idx = int(len(df) * 0.8) # 80% Train, 20% Val
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Configurar Tokenizer y Text Data para LoRA
    tokenizer = None
    text_data = None
    
    # Si usamos LoRA, necesitamos el texto crudo, no los embeddings pre-computados
    # Asumimos que el CSV tiene una columna 'lyrics' o similar
    # Para simplificar, extraemos el texto del DF original
    print("Preparando datos de texto para LoRA...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False)
    
    # Crear diccionario track_id -> texto
    # Prioridad: lyrics > artist - track
    if 'lyrics' in df.columns:
        text_df = df[['track_id', 'lyrics']].drop_duplicates('track_id')
        text_df['lyrics'] = text_df['lyrics'].fillna("")
        text_data = dict(zip(text_df['track_id'], text_df['lyrics']))
    else:
        print("Usando metadata (Artist - Track) como texto fallback")
        unique_tracks = df.drop_duplicates('track_id')
        texts = (unique_tracks['artist_name'].fillna("") + " - " + 
                 unique_tracks['track_name'].fillna("") + " (" + 
                 unique_tracks['album_name'].fillna("") + ")")
        text_data = dict(zip(unique_tracks['track_id'], texts))

    # Datasets
    print("Inicializando Datasets...")
    train_dataset = MultimodalDataset(
        interactions_df=train_df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        audio_dir=args.audio_dir,
        text_data=text_data,
        tokenizer=tokenizer
    )
    
    # Para validación, necesitamos la historia completa (incluyendo lo visto en train)
    # MultimodalDataset construye la historia desde el DF que se le pasa.
    # Truco: Pasamos val_df para los targets, pero inyectamos la historia completa manualmente.
    val_dataset = MultimodalDataset(
        interactions_df=val_df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        audio_dir=args.audio_dir,
        text_data=text_data,
        tokenizer=tokenizer,
        encoders=train_dataset.get_encoders() # Usar mismos scalers
    )
    
    # Inyectar historia completa al dataset de validación
    full_user_groups = df.groupby('user_id')['track_id'].apply(list).to_dict()
    val_dataset.user_groups = full_user_groups
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Modelo
    vocab_size = len(item_id_mapper) + 1
    tabular_dim = train_dataset.tabular_data.shape[1]
    
    # Obtener dimensiones de atributos de usuario
    # Si por alguna razón no se crearon (ej. dataset vacío), fallback a 1
    num_genders = len(train_dataset.encoders['gender_encoder'].classes_) if 'gender_encoder' in train_dataset.encoders else 1
    num_countries = len(train_dataset.encoders['country_encoder'].classes_) if 'country_encoder' in train_dataset.encoders else 1

    print(f"Vocab Size: {vocab_size}, Tabular Dim: {tabular_dim}, Genders: {num_genders}, Countries: {num_countries}")
    
    model = TwoTowerModel(
        vocab_size=vocab_size,
        num_genders=num_genders,
        num_countries=num_countries,
        tabular_input_dim=tabular_dim,
        item_embedding_dim=256,
        user_embedding_dim=256,
        use_lora=True # Activar LoRA
    ).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    best_recall = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Iniciando entrenamiento...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device, epoch+1)
        val_recall = evaluate(model, val_loader, args.device, k=10)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Recall@10 (Batch): {val_recall:.4f}")
        
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Saved best model with Recall@10: {best_recall:.4f}")
            
    print("Entrenamiento finalizado.")

if __name__ == '__main__':
    main()
