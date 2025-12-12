import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from tqdm import tqdm
import json
import argparse
import logging
import joblib
from transformers import AutoTokenizer

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

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, device, epoch, is_main_process=True, log_interval=50):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()

    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        # Mover batch a dispositivo
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad(set_to_none=True) # Más eficiente que zero_grad()

        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            # Solo necesitamos loss, ignoramos el resto para ahorrar memoria si es posible
            # (aunque el grafo ya se creó, evitamos mantener referencias extra)
            loss, _, _, _ = model(batch)

        # Backward pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val

        if is_main_process and (i + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch} [{i+1}/{num_batches}] | Loss: {loss_val:.4f}")

        # Liberar memoria explícitamente
        del loss, batch

    return total_loss / num_batches

def evaluate(model, dataloader, device, k=10):
    model.eval()
    # Evaluación simplificada: Recall@K "in-batch"
    # Verificamos si el item positivo está en el top-K de items del batch para ese usuario.

    hits = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not (dist.get_rank() == 0)):
            # Mover tensores al dispositivo
            for k_key, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k_key] = v.to(device)

            # Nota: Usamos el modelo sin DDP wrapper (model.module) si es posible para evitar overhead,
            # pero si pasas el modelo DDP directo, asegúrate de que esté en .eval() para no sincronizar grads.
            _, logits, _, _ = model(batch)

            # --- Lógica de métricas (igual que antes) ---
            _, topk_indices = torch.topk(logits, k=k, dim=1)
            targets = torch.arange(logits.shape[0], device=device).unsqueeze(1)
            is_hit = (topk_indices == targets).any(dim=1)

            hits += is_hit.sum()
            total += logits.shape[0]

    # --- SINCRONIZACIÓN DDP ---
    if dist.is_initialized():
        # Sumamos los hits y el total de todas las GPUs
        dist.all_reduce(hits, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    return (hits / total).item() if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Multimodal Model")
    parser.add_argument("--data_path", type=str, default="data/spotify-kaggle/interim/lastfm_spotify_merged.csv")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/")
    parser.add_argument("--audio_dir", type=str, default="data/audio/mels/", help="Path to audio embeddings dir")
    parser.add_argument("--lyrics_path", type=str, default="data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv", help="Path to lyrics CSV file")
    parser.add_argument("--mapper_path", type=str, default="data/spotify-kaggle/interim/item_id_mapper.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device # Override device arg
    is_main_process = (local_rank == 0)

    if is_main_process:
        logger.info(f"Using device: {device}")

    # 1. Cargar Datos
    if is_main_process:
        logger.info("Cargando datos...")
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found at {args.data_path}")
        raise FileNotFoundError(f"Data file not found at {args.data_path}")

    df = pd.read_csv(args.data_path)

    # Cargar o Crear Mapper
    if os.path.exists(args.mapper_path):
        logger.info(f"Cargando mapper desde {args.mapper_path}")
        with open(args.mapper_path, 'r') as f:
            item_id_mapper = json.load(f)
    else:
        logger.info("Mapper no encontrado, creando desde datos...")
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

    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Configurar Tokenizer y Text Data para LoRA
    tokenizer = None
    text_data = None

    # Si usamos LoRA, necesitamos el texto crudo, no los embeddings pre-computados
    logger.info("Preparando datos de texto para LoRA...")
    cache_dir = os.getenv('HF_HOME')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False, cache_dir=cache_dir)

    # 1. Generar texto base (Metadata) para TODOS los tracks
    # Esto asegura que siempre haya algo de texto (Artist - Track)
    logger.info("Generando texto base desde metadata...")
    unique_tracks = df.drop_duplicates('track_id')
    texts = (unique_tracks['artist_name'].fillna("") + " - " +
             unique_tracks['track_name'].fillna("") + " (" +
             unique_tracks['album_name'].fillna("") + ")")
    text_data = dict(zip(unique_tracks['track_id'], texts))

    # 2. Cargar Lyrics si se proporcionan y actualizar el diccionario
    if args.lyrics_path and os.path.exists(args.lyrics_path):
        logger.info(f"Cargando lyrics desde {args.lyrics_path}...")
        try:
            lyrics_df = pd.read_csv(args.lyrics_path)
            if 'track_id' in lyrics_df.columns and 'lyrics' in lyrics_df.columns:
                # Filtrar lyrics vacíos o muy cortos si es necesario
                lyrics_df = lyrics_df.dropna(subset=['lyrics'])
                lyrics_df = lyrics_df[lyrics_df['lyrics'].str.strip().str.len() > 0]

                # Crear dict de lyrics
                lyrics_dict = dict(zip(lyrics_df['track_id'], lyrics_df['lyrics']))

                # Actualizar text_data (sobrescribe metadata con lyrics donde haya match)
                # Opcional: Concatenar metadata + lyrics?
                # Por ahora, reemplazamos metadata con lyrics porque es más rico.
                count_updates = 0
                for tid, lyric in lyrics_dict.items():
                    if tid in text_data:
                        text_data[tid] = lyric # Reemplazar
                        count_updates += 1
                logger.info(f"Lyrics actualizados para {count_updates} tracks.")
            else:
                logger.warning("Advertencia: El CSV de lyrics no tiene columnas 'track_id' y 'lyrics'.")
        except Exception as e:
            logger.error(f"Error cargando lyrics: {e}")
    elif 'lyrics' in df.columns:
        # Fallback legacy: si el DF principal ya tenía lyrics
        logger.info("Usando columna 'lyrics' del dataset principal...")
        text_df = df[['track_id', 'lyrics']].drop_duplicates('track_id')
        text_df = text_df.dropna(subset=['lyrics'])
        lyrics_dict = dict(zip(text_df['track_id'], text_df['lyrics']))
        text_data.update(lyrics_dict)

    # Datasets
    logger.info("Inicializando Datasets...")
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

    # Guardar Encoders para Inferencia
    encoders_path = os.path.join(os.path.dirname(args.mapper_path), 'encoders.pkl')
    logger.info(f"Guardando encoders en {encoders_path}...")
    joblib.dump(train_dataset.encoders, encoders_path)

    # DataLoaders
    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True, # Mantiene workers vivos entre épocas
        prefetch_factor=2        # Pre-carga 2 batches por worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 2. Modelo
    vocab_size = len(item_id_mapper) + 1
    tabular_dim = train_dataset.tabular_data.shape[1]

    # Obtener dimensiones de atributos de usuario
    # Si por alguna razón no se crearon (ej. dataset vacío), fallback a 1
    num_genders = len(train_dataset.encoders['gender_encoder'].classes_) if 'gender_encoder' in train_dataset.encoders else 1
    num_countries = len(train_dataset.encoders['country_encoder'].classes_) if 'country_encoder' in train_dataset.encoders else 1

    logger.info(f"Vocab Size: {vocab_size}, Tabular Dim: {tabular_dim}, Genders: {num_genders}, Countries: {num_countries}")

    model = TwoTowerModel(
        vocab_size=vocab_size,
        num_genders=num_genders,
        num_countries=num_countries,
        tabular_input_dim=tabular_dim,
        item_embedding_dim=256,
        user_embedding_dim=256,
        use_lora=True # Activar LoRA
    ).to(args.device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 3. Training Loop
    best_recall = 0.0
    if is_main_process:
        os.makedirs("checkpoints", exist_ok=True)
        logger.info("Iniciando entrenamiento...")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) # Important for shuffling in DDP
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device, epoch+1, is_main_process)

        # Para evaluación, usamos el modelo base (sin DDP wrapper)
        # En DDP, model.module es el modelo original
        if isinstance(model, DDP):
            eval_model = model.module
        else:
            eval_model = model

        val_recall = evaluate(eval_model, val_loader, args.device, k=10)

        # Logging y Guardado (Solo Rank 0 imprime y guarda)
        if is_main_process:
            logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Recall@10 (Batch): {val_recall:.4f}")

            if val_recall > best_recall:
                best_recall = val_recall
                torch.save(eval_model.state_dict(), f'checkpoints/best_model_epoch{epoch+1}.pth')
                logger.info(f"Saved best model with Recall@10: {best_recall:.4f}")

        # CRITICAL FIX: Wait for main process to finish evaluation before starting next epoch
        if dist.is_initialized():
            dist.barrier()

        # Limpiar caché de GPU al final de la época para evitar fragmentación
        torch.cuda.empty_cache()

    if is_main_process:
        logger.info("Entrenamiento finalizado.")

    cleanup_ddp()

if __name__ == '__main__':
    main()
