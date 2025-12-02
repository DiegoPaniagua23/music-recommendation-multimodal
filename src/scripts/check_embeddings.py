import torch
import os
import sys

def check_embeddings(file_path):
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return

    print(f"--- Verificando {file_path} ---")
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"Error al cargar el archivo con torch.load: {e}")
        return

    if not isinstance(data, dict):
        print(f"Formato incorrecto: Se esperaba un diccionario, se obtuvo {type(data)}")
        return

    num_items = len(data)
    print(f"Total de items (canciones): {num_items}")

    if num_items == 0:
        print("Advertencia: El diccionario está vacío.")
        return

    # Inspeccionar una muestra
    sample_key = next(iter(data))
    sample_emb = data[sample_key]

    print(f"\nMuestra (Track ID: {sample_key}):")
    print(f"  - Tipo: {type(sample_emb)}")
    
    if isinstance(sample_emb, torch.Tensor):
        print(f"  - Shape: {sample_emb.shape}")
        print(f"  - Dtype: {sample_emb.dtype}")
        print(f"  - Device: {sample_emb.device}")
        
        # Chequeos de valores
        has_nan = torch.isnan(sample_emb).any().item()
        has_inf = torch.isinf(sample_emb).any().item()
        is_zero = (sample_emb == 0).all().item()
        
        print(f"  - Contiene NaNs: {has_nan}")
        print(f"  - Contiene Infs: {has_inf}")
        print(f"  - Es todo ceros: {is_zero}")
        
        if not has_nan and not has_inf and not is_zero:
            print("\n✅ Integridad de la muestra: CORRECTA")
            if sample_emb.shape[0] == 768:
                print("✅ Dimensiones correctas para mDeBERTa-v3-base (768)")
            else:
                print(f"⚠️ Dimensiones inesperadas (se esperaba 768, se obtuvo {sample_emb.shape[0]})")
        else:
            print("\n❌ PROBLEMAS DE INTEGRIDAD DETECTADOS")
    else:
        print("❌ El valor no es un tensor de PyTorch")

if __name__ == "__main__":
    # Ruta por defecto usada en el script de generación
    default_path = "./data/spotify-kaggle/processed/lyrics_embeddings_10k_fixed.pt"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    check_embeddings(path)
