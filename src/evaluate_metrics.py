import argparse
import logging
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import MultimodalDataset
from src.models import TwoTowerModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_all_item_embeddings(model, dataset, batch_size, device):
    """
    Genera embeddings para TODOS los items únicos en el dataset.
    Retorna:
        item_embeddings: Tensor (Num_Items, Embedding_Dim)
        id_to_idx: Mapping de track_id -> índice en el tensor
    """
    logger.info("Generando índice de items (Catálogo completo)...")
    
    # 1. Crear un dataset de items únicos
    unique_df = dataset.interactions_df.drop_duplicates('track_id').copy()
    # Dummy user info para que pase el __getitem__
    unique_df['user_id'] = 'dummy_user'
    unique_df['timestamp'] = '2020-01-01'
    
    # Reutilizamos la clase MultimodalDataset pero con el DF único
    catalog_dataset = MultimodalDataset(
        interactions_df=unique_df,
        item_id_mapper=dataset.item_id_mapper,
        img_dir=dataset.img_dir,
        audio_dir=dataset.audio_dir,
        text_data=dataset.text_data,
        tokenizer=dataset.tokenizer,
        encoders=dataset.encoders # Usar encoders ya ajustados
    )
    
    loader = DataLoader(catalog_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    embeddings_list = []
    track_ids_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Indexing Items"):
            # Guardar track_ids para saber el orden
            # batch['target_id'] son los enteros mapeados
            track_ids_list.append(batch['target_id'])
            
            # Forward Item Tower
            emb = model.get_item_embedding(
                images=batch['target_image'].to(device),
                audio=batch['target_audio'].to(device) if batch['target_audio'] is not None else None,
                input_ids=batch['target_input_ids'].to(device),
                attention_mask=batch['target_attention_mask'].to(device),
                tabular=batch['target_tabular'].to(device)
            )
            embeddings_list.append(emb.cpu())
            
    # Concatenar
    all_embeddings = torch.cat(embeddings_list, dim=0) # (M, D)
    all_ids = torch.cat(track_ids_list, dim=0) # (M,)
    
    # Crear matriz densa donde el índice corresponde al item_id entero
    # El item_id_mapper va de 1 a V. El 0 es padding.
    vocab_size = len(dataset.item_id_mapper) + 1
    embedding_dim = all_embeddings.shape[1]
    
    # Matriz final ordenada por ID (índice 0 es padding/dummy)
    dense_embeddings = torch.zeros(vocab_size, embedding_dim)
    
    # Asignar embeddings a sus posiciones correctas
    # all_ids contiene los índices enteros del mapper
    dense_embeddings[all_ids.long()] = all_embeddings
    
    return dense_embeddings, vocab_size

def calculate_metrics_global(model, val_loader, item_embeddings, device, k_list=[10, 20]):
    """
    Evalúa el modelo recuperando items del catálogo completo (Global Retrieval).
    """
    logger.info(f"Iniciando evaluación global con K={k_list}...")
    
    model.eval()
    item_embeddings = item_embeddings.to(device) # (Vocab_Size, D)
    
    metrics = {f"Recall@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Users"):
            # 1. Obtener Embedding del Usuario
            user_emb = model.get_user_embedding(
                history_ids=batch['history_ids'].to(device),
                history_mask=batch['history_mask'].to(device),
                user_gender=batch['user_gender'].to(device),
                user_country=batch['user_country'].to(device)
            ) # (B, D)
            
            # 2. Calcular Scores contra TODO el catálogo
            # (B, D) @ (V, D).T -> (B, V)
            # Esto puede ser pesado en memoria si V es muy grande (>1M). 
            # Para datasets medianos (<100k) cabe en GPU moderna.
            scores = torch.matmul(user_emb, item_embeddings.t())
            
            # 3. Enmascarar Padding (índice 0) y el propio historial (opcional)
            # Por ahora solo enmascaramos el padding (índice 0)
            scores[:, 0] = -float('inf')
            
            # 4. Obtener Top-K predicciones
            max_k = max(k_list)
            _, topk_indices = torch.topk(scores, k=max_k, dim=1) # (B, Max_K)
            
            # 5. Comparar con Ground Truth
            targets = batch['target_id'].to(device).unsqueeze(1) # (B, 1)
            
            # Calcular métricas para cada K
            for k in k_list:
                # Cortar a K
                preds_k = topk_indices[:, :k]
                
                # Hit: ¿Está el target en las predicciones?
                hits = (preds_k == targets).any(dim=1) # (B,)
                metrics[f"Recall@{k}"].append(hits.float().cpu())
                
                # NDCG Calculation
                # Encontrar la posición del hit (si existe)
                # (preds_k == targets) es una matriz booleana (B, K) con un True donde está el hit
                hit_positions = (preds_k == targets).nonzero(as_tuple=False)
                
                ndcg_scores = torch.zeros(hits.shape[0], device=device)
                if hit_positions.shape[0] > 0:
                    # hit_positions[:, 1] es el rango (0-indexed)
                    ranks = hit_positions[:, 1]
                    # DCG = 1 / log2(rank + 2)
                    gains = 1.0 / torch.log2(ranks.float() + 2.0)
                    # Asignar gain al batch correspondiente
                    batch_indices = hit_positions[:, 0]
                    ndcg_scores[batch_indices] = gains
                
                metrics[f"NDCG@{k}"].append(ndcg_scores.cpu())

    # Promediar resultados
    final_results = {}
    for name, values in metrics.items():
        final_results[name] = torch.cat(values).mean().item()
        
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower Model with Global Metrics")
    parser.add_argument("--data_path", type=str, default="data/spotify-kaggle/interim/lastfm_spotify_merged.csv")
    parser.add_argument("--mapper_path", type=str, default="data/spotify-kaggle/interim/item_id_mapper.json")
    parser.add_argument("--img_dir", type=str, default="data/spotify-kaggle/album_covers/")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # 1. Cargar Datos (Igual que en train.py para consistencia)
    logger.info("Cargando datos y recursos...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}")
        
    df = pd.read_csv(args.data_path)
    
    # Cargar Mapper
    with open(args.mapper_path, 'r') as f:
        item_id_mapper = json.load(f)
        
    # Preparar Texto (Metadata básica)
    logger.info("Preparando metadatos de texto...")
    cache_dir = os.getenv('HF_HOME')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False, cache_dir=cache_dir)
    
    unique_tracks = df.drop_duplicates('track_id')
    texts = (unique_tracks['artist_name'].fillna("") + " - " + 
             unique_tracks['track_name'].fillna("") + " (" + 
             unique_tracks['album_name'].fillna("") + ")")
    text_data = dict(zip(unique_tracks['track_id'], texts))
    
    # 2. Split Train/Val (Misma lógica que train.py)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    # 3. Inicializar Datasets
    # Necesitamos instanciar el de Train primero SOLO para ajustar los encoders (StandardScaler, etc)
    logger.info("Ajustando encoders con datos de entrenamiento...")
    train_dataset = MultimodalDataset(
        interactions_df=train_df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        text_data=text_data,
        tokenizer=tokenizer
    )
    
    # Dataset de Validación (Evaluación)
    # Inyectamos la historia completa
    full_user_groups = df.groupby('user_id')['track_id'].apply(list).to_dict()
    
    val_dataset = MultimodalDataset(
        interactions_df=val_df,
        item_id_mapper=item_id_mapper,
        img_dir=args.img_dir,
        text_data=text_data,
        tokenizer=tokenizer,
        encoders=train_dataset.get_encoders() # IMPORTANTE: Usar mismos encoders
    )
    val_dataset.user_groups = full_user_groups
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 4. Cargar Modelo
    logger.info("Cargando modelo...")
    vocab_size = len(item_id_mapper) + 1
    tabular_dim = train_dataset.tabular_data.shape[1]
    
    # Obtener dimensiones reales de los encoders
    num_genders = len(train_dataset.encoders['gender_encoder'].classes_)
    num_countries = len(train_dataset.encoders['country_encoder'].classes_)
    
    model = TwoTowerModel(
        vocab_size=vocab_size,
        num_genders=num_genders,
        num_countries=num_countries,
        tabular_input_dim=tabular_dim,
        item_embedding_dim=256,
        user_embedding_dim=256,
        use_lora=True
    )
    
    # Cargar pesos
    state_dict = torch.load(args.model_path, map_location=device)
    # Si se guardó con DDP, las keys tienen prefijo 'module.'
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    
    # 5. Generar Embeddings de Items (Index)
    item_embeddings, vocab_size_idx = compute_all_item_embeddings(model, train_dataset, args.batch_size, device)
    
    # 6. Calcular Métricas
    results = calculate_metrics_global(model, val_loader, item_embeddings, device, k_list=[10, 20, 50])
    
    print("\n" + "="*30)
    print("RESULTADOS DE EVALUACIÓN GLOBAL")
    print("="*30)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    main()
