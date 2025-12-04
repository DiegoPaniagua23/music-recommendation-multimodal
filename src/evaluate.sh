#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=RS_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/logs/evaluation.log"
#SBATCH --error="/home/est_posgrado_gustavo.angeles/Proyectos/music-recommendation-multimodal/logs/evaluation.err"
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

PYTHON_SCRIPT="./src/evaluate_metrics.py"
MODEL_CHECKPOINT="checkpoints/complete/best_model_epoch1.pth"

echo "Ejecutando evaluación con script: $PYTHON_SCRIPT"
echo "Usando checkpoint: $MODEL_CHECKPOINT"

# Usar 'uv run' para asegurar que se usen las dependencias del proyecto
# No usamos torchrun porque la evaluación no necesita DDP (es single GPU)
uv run python "$PYTHON_SCRIPT" \
    --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
    --img_dir "data/spotify-kaggle/album_covers/" \
    --audio_dir "data/audio/mels/" \
    --model_path "$MODEL_CHECKPOINT" \
    --batch_size 64

if [ $? -ne 0 ]; then
    echo "ERROR: El script de evaluación falló."
    exit 1
fi

echo "--- Trabajo de Evaluación Finalizado ---"
