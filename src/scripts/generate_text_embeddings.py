import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os

def generate_embeddings(data_path, output_path, model_name="microsoft/mdeberta-v3-base", batch_size=32, device="cuda"):
    print(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)
    
    # Verificar si existe columna de lyrics o texto
    # Asumimos que hay una columna 'lyrics' o combinamos 'artist_name' + 'track_name' como fallback
    if 'lyrics' in df.columns:
        print("Usando columna 'lyrics' para embeddings.")
        # Filtrar canciones únicas con lyrics
        unique_tracks = df.drop_duplicates(subset=['track_id'])
        texts = unique_tracks['lyrics'].fillna("").tolist()
        track_ids = unique_tracks['track_id'].tolist()
    else:
        print("ADVERTENCIA: No se encontró columna 'lyrics'. Usando 'artist_name' + 'track_name' + 'album_name'.")
        unique_tracks = df.drop_duplicates(subset=['track_id'])
        texts = (unique_tracks['artist_name'].fillna("") + " - " + 
                 unique_tracks['track_name'].fillna("") + " (" + 
                 unique_tracks['album_name'].fillna("") + ")").tolist()
        track_ids = unique_tracks['track_id'].tolist()
        
    print(f"Generando embeddings para {len(texts)} items únicos...")
    
    # Cargar Modelo y Tokenizer
    # use_fast=False es necesario para mDeBERTa-v3 para evitar errores de conversión con protobuf/sentencepiece
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings_dict = {}
    
    # Procesar en batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_ids = track_ids[i:i+batch_size]
        
        # Tokenizar
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Usar el embedding del token [CLS] (primer token) o Mean Pooling
            # mDeBERTa v3 usa Mean Pooling usualmente para sentence embeddings, pero CLS funciona bien.
            # Aquí usamos Mean Pooling de la última capa oculta para mejor representación semántica.
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
            # Mover a CPU
            batch_embeddings = batch_embeddings.cpu()
            
        # Guardar en diccionario
        for tid, emb in zip(batch_ids, batch_embeddings):
            embeddings_dict[tid] = emb
            
    # Guardar
    print(f"Guardando embeddings en {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_dict, output_path)
    print("¡Listo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv", help="Path al CSV con datos")
    parser.add_argument("--output_path", type=str, default="./data/spotify-kaggle/processed/lyrics_embeddings_10k_fixed.pt", help="Path para guardar .pt")
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    generate_embeddings(args.data_path, args.output_path, args.model_name, args.batch_size, args.device)
