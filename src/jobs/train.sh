#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=RS_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/logs/training.log"
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/logs/training.err"
#SBATCH --mem=0
#SBATCH --time=0

# Enviar correo electrónico cuando el trabajo finalice o falle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx

####################### CONFIGURACION DE UV #######################

# add uv to path
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Checking available GPUs..."
nvidia-smi


# Configurar cache de modelos (para nodo sin internet)
export TORCH_HOME="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/model_cache/torch"
export HF_HOME="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/model_cache/huggingface"

# Forzar modo offline para Hugging Face
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# change to the directory where the job was submitted
cd "/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal"

# Agregar el directorio actual al PYTHONPATH para que Python encuentre el módulo 'src'
export PYTHONPATH=$PWD

####################### EJECUCION DEL SCRIPT DE PYTHON #######################

PYTHON_SCRIPT="./src/train.py"

echo "Ejecutando el script de Python: $PYTHON_SCRIPT"

# Usar 'uv run' para asegurar que se usen las dependencias del proyecto
# Usar torchrun para DDP
uv run torchrun --nproc_per_node=2 "$PYTHON_SCRIPT" \
    --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
    --img_dir "data/spotify-kaggle/album_covers/" \
    --lyrics_path "data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv" \
    --text_embeddings_path "data/spotify-kaggle/processed/lyrics_embeddings_10k_fixed.pt" \
    --epochs 10 \
    --batch_size 32

if [ $? -ne 0 ]; then
    echo "ERROR: El script de Python falló."
    exit 1
fi

echo "--- Trabajo de Slurm Finalizado ---"
