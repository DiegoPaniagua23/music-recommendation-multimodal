import os
import torch
from torchvision import models
from transformers import AutoTokenizer, AutoModel

def download_models(cache_dir="./model_cache"):
    print(f"Descargando modelos a {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Configurar variables de entorno para cache
    os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')
    os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    
    print("1. Descargando ResNet18 (ImageNet weights)...")
    # Esto descargará los pesos a TORCH_HOME/hub/checkpoints
    try:
        models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("   -> ResNet18 descargado exitosamente.")
    except Exception as e:
        print(f"   -> Error descargando ResNet18: {e}")

    print("2. Descargando mDeBERTa v3 base...")
    model_name = "microsoft/mdeberta-v3-base"
    try:
        # Descargar Tokenizer y Modelo
        AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ['HF_HOME'])
        AutoModel.from_pretrained(model_name, cache_dir=os.environ['HF_HOME'])
        print(f"   -> {model_name} descargado exitosamente.")
    except Exception as e:
        print(f"   -> Error descargando {model_name}: {e}")

    print("\nDescarga completada.")
    print(f"Recuerda configurar las variables de entorno en tu script de training si usas este cache personalizado:")
    print(f"export TORCH_HOME={os.path.abspath(os.path.join(cache_dir, 'torch'))}")
    print(f"export HF_HOME={os.path.abspath(os.path.join(cache_dir, 'huggingface'))}")

if __name__ == "__main__":
    download_models()
