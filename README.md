# 🎵 Arquitectura Híbrida Two-Tower para Recomendación Musical Multimodal

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-purple)
![Status](https://img.shields.io/badge/Status-Development-yellow)

---

## 📖 Descripción General

Este proyecto implementa un sistema de recomendación musical del Estado del Arte (SOTA) utilizando una arquitectura **Two-Tower** con fusión **Cross-Modal**. El objetivo es resolver los problemas de escasez de datos (*sparsity*) y brecha semántica (*semantic gap*) en los sistemas tradicionales.

El modelo alinea dos espacios vectoriales:
1.  **User Tower:** Codifica la secuencia histórica de interacciones del usuario usando **SASRec** (Transformer secuencial).
2.  **Item Tower:** Codifica el contenido de la canción mediante **Atención Cruzada (Cross-Attention)** entre Audio (Mel-Spectrograms), Texto (Lyrics) e Imagen (Carátulas).

## 🏗️ Arquitectura del Sistema


## 🚀 Instalación y Configuración

Este proyecto utiliza **`uv`** para la gestión de dependencias y **DVC** para el control de versiones de datos.

### Prerrequisitos

  * Python 3.9+
  * [uv](https://github.com/astral-sh/uv) instalado.
  * `ffmpeg` instalado en el sistema (para procesamiento de audio).

### 1\. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-mir
```

### 2\. Instalar dependencias

```bash
uv sync
# Esto creará el entorno virtual e instalará todo lo necesario
```

### 3. Configurar Datos (DVC)

Para descargar los datos, necesitas configurar las credenciales de Google Drive. Ejecuta los siguientes comandos:

```bash
# Configurar credenciales locales (no se suben al repo, se adjuntan en el trabajo)
dvc remote modify --local proyecto_multimodal gdrive_client_id 
"<gdrive_client_id>"
dvc remote modify --local proyecto_multimodal gdrive_client_secret 
"<gdrive_client_secret>"

# Descargar datos
uv run dvc pull
```

### 4. Variables de Entorno

Asegúrate de tener configuradas las variables necesarias, especialmente si usas modelos de HuggingFace que requieran token (aunque mDeBERTa es público).

```bash
export HF_HOME="./.cache/huggingface"
```

-----

## 🏋️‍♂️ Entrenamiento

Para entrenar el modelo desde cero hemos utilizado el clúster del CIMAT (Bajío), el cual cuenta con 2 GPUs en cada nodo. El funcionamiento puede variar dependiendo del hardware donde se quiera reproducir el entrenamiento (Los requerimientos de Hardware son altos, un tamaño de lote de 64 requiere mas de 24 GB VRAM). 

En todo caso, utilizamos el script `src/train.py`. Este script se encarga de:
1. Cargar y preprocesar los datos.
2. Ajustar y guardar los encoders (necesarios para inferencia).
3. Entrenar el modelo Two-Tower.

```bash
# Agregar el directorio actual al PYTHONPATH para que Python encuentre el módulo 'src'
export PYTHONPATH=$PWD

# Para el clúster CIMAT utilizamos el script train.sh
uv run python -m torch.distributed.run --nproc_per_node=1 src/train.py \
    --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
    --img_dir "data/spotify-kaggle/album_covers/" \
    --audio_dir "data/audio/mels/" \
    --lyrics_path "data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv" \
    --epochs 10 \
    --batch_size 32

# En local (considerando una GPU).
uv run python -m torch.distributed.run --nproc_per_node=1 src/train.py \
    --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
    --img_dir "data/spotify-kaggle/album_covers/" \
    --audio_dir "data/audio/mels/" \
    --lyrics_path "data/spotify-kaggle/interim/lyrics_dataset_10k_fixed.csv" \
    --epochs 10 \
    --batch_size 32
```

**Nota:** Los checkpoints y encoders se guardarán automáticamente en la carpeta `checkpoints/`.

-----

## 📊 Evaluación

Para evaluar el rendimiento del modelo (Recall@K, NDCG@K) sobre el conjunto de validación/test:

```bash
uv run python src/evaluate_metrics.py \
  --model_path "checkpoints/complete/best_model_epoch1.pth" \
  --encoders_path "checkpoints/complete/encoders.pkl" \
  --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
  --embeddings_cache_path "checkpoints/complete/item_embeddings_cache_epoch1.pt" \
  --batch_size 32
```

-----

## 🔮 Inferencia y Recomendación

El sistema de inferencia tiene dos modos: **Indexación** y **Recomendación**.

### Paso 1: Indexación (`index`)
Pre-calcula los embeddings de todas las canciones del catálogo para una búsqueda rápida.

```bash
uv run python -m src.inference \
  --mode index \
  --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
  --mapper_path "data/spotify-kaggle/interim/item_id_mapper.json" \
  --model_path "checkpoints/complete/best_model_epoch1.pth" \
  --encoders_path "checkpoints/complete/encoders.pkl" \
  --index_path "checkpoints/item_index_epoch1.pt"
```

### Paso 2: Recomendación (`recommend`)
Genera recomendaciones personalizadas para un usuario específico basándose en su historial.

```bash
uv run python -m src.inference \
  --mode recommend \
  --user_id "user_000238" \
  --data_path "data/spotify-kaggle/interim/lastfm_spotify_merged.csv" \
  --mapper_path "data/spotify-kaggle/interim/item_id_mapper.json" \
  --model_path "checkpoints/best_model.pth" \
  --encoders_path "checkpoints/encoders.pkl" \
  --index_path "checkpoints/item_index.pt"
```

-----

## 📂 Estructura del Proyecto

```
.
├── data/               # Datos crudos y procesados (gestionado por DVC)
├── notebooks/          # Jupyter Notebooks para EDA y prototipado
├── src/                # Código fuente
│   ├── dataset.py      # Clase MultimodalDataset y lógica de carga
│   ├── models/         # Definición de arquitecturas (TwoTower, Encoders)
│   ├── train.py        # Script de entrenamiento
│   ├── inference.py    # Script de inferencia y recomendación
│   └── evaluate_metrics.py # Script de evaluación
├── checkpoints/        # Modelos entrenados y encoders guardados
├── pyproject.toml      # Dependencias y configuración del proyecto
└── uv.lock             # Lockfile de dependencias
```

-----

## 🤝 Flujo de Trabajo Colaborativo



-----

## 👥 Equipo y Roles



## 📜 Licencia


