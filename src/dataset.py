import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

class MultimodalDataset(Dataset):
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_id_mapper: Dict[str, int],
        img_dir: str,
        audio_dir: Optional[str] = None,
        text_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        max_seq_len: int = 30,
        transform: Optional[transforms.Compose] = None,
        encoders: Optional[Dict[str, Any]] = None
    ):
        """
        Dataset multimodal para arquitectura Two-Tower.

        Args:
            interactions_df (pd.DataFrame): DataFrame con interacciones.
            item_id_mapper (Dict[str, int]): Mapeo de track_id a índice entero.
            img_dir (str): Directorio con las imágenes de portadas.
            audio_dir (str, optional): Directorio con tensores de Mel Spectrograms.
            text_embeddings (Dict[str, torch.Tensor], optional): Embeddings de texto pre-cargados.
            max_seq_len (int): Longitud máxima de la secuencia histórica.
            transform (transforms.Compose, optional): Transformaciones para imágenes.
            encoders (Dict[str, Any], optional): Diccionario con encoders ya ajustados (StandardScaler, OneHotEncoder, LabelEncoder).
                                                 Si es None, se ajustarán con los datos provistos.
        """
        self.interactions_df = interactions_df.copy()
        self.item_id_mapper = item_id_mapper
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.text_embeddings = text_embeddings
        self.max_seq_len = max_seq_len
        self.encoders = encoders if encoders else {}

        # 1. Transformaciones de Imagen
        # Se utiliza 224x224 y la normalización estándar de ImageNet porque utilizaremos
        # una ResNet pre-entrenada como backbone visual. Estos modelos esperan esta entrada.
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 2. Procesamiento de Features Tabulares
        self.numerical_features = [
            'popularity', 'danceability', 'energy', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
        ]

        # Filtrar columnas que realmente existen en el DF
        self.numerical_features = [c for c in self.numerical_features if c in self.interactions_df.columns]

        # A. Scaling de variables numéricas (StandardScaler)
        if 'scaler' not in self.encoders:
            self.encoders['scaler'] = StandardScaler()
            self.interactions_df[self.numerical_features] = self.encoders['scaler'].fit_transform(
                self.interactions_df[self.numerical_features]
            )
        else:
            self.interactions_df[self.numerical_features] = self.encoders['scaler'].transform(
                self.interactions_df[self.numerical_features]
            )

        # B. One-Hot Encoding de Género (OneHotEncoder de sklearn)
        self.genre_features = []
        if 'track_genre' in self.interactions_df.columns:
            # Reshape necesario para sklearn (N, 1)
            genres = self.interactions_df[['track_genre']].values

            if 'genre_encoder' not in self.encoders:
                self.encoders['genre_encoder'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                genre_encoded = self.encoders['genre_encoder'].fit_transform(genres)
            else:
                genre_encoded = self.encoders['genre_encoder'].transform(genres)

            # Nombres de las nuevas columnas
            self.genre_features = [f"genre_{cat}" for cat in self.encoders['genre_encoder'].categories_[0]]

            # Concatenar al DF (usando un DataFrame temporal para evitar fragmentación)
            genre_df = pd.DataFrame(genre_encoded, columns=self.genre_features, index=self.interactions_df.index)
            self.interactions_df = pd.concat([self.interactions_df, genre_df], axis=1)

        # Lista final de features tabulares
        self.tabular_features = self.numerical_features + self.genre_features

        # C. Encoding de Features de Usuario (Gender, Country)
        # Rellenar nulos
        if 'gender' in self.interactions_df.columns:
            self.interactions_df['gender'] = self.interactions_df['gender'].fillna('unknown')
        else:
            self.interactions_df['gender'] = 'unknown'

        if 'country' in self.interactions_df.columns:
            self.interactions_df['country'] = self.interactions_df['country'].fillna('unknown')
        else:
            self.interactions_df['country'] = 'unknown'

        # Gender Encoder
        if 'gender_encoder' not in self.encoders:
            self.encoders['gender_encoder'] = LabelEncoder()
            # Ajustar con todos los valores posibles + 'unknown'
            self.encoders['gender_encoder'].fit(self.interactions_df['gender'])

        # Country Encoder
        if 'country_encoder' not in self.encoders:
            self.encoders['country_encoder'] = LabelEncoder()
            self.encoders['country_encoder'].fit(self.interactions_df['country'])

        # Transformar y guardar en columnas numéricas
        # Usamos transform para asegurar consistencia (si hay valores nuevos en val, LabelEncoder fallará si no se maneja)
        # Para robustez en producción, deberíamos manejar 'unknown' para valores no vistos.
        # Aquí asumimos que 'unknown' ya cubre nulos, pero si aparece un país nuevo en val, LabelEncoder lanzará error.
        # Solución rápida: Mapear valores desconocidos a 'unknown' antes de transform.

        known_genders = set(self.encoders['gender_encoder'].classes_)
        self.interactions_df['gender'] = self.interactions_df['gender'].apply(lambda x: x if x in known_genders else 'unknown')
        self.interactions_df['gender_idx'] = self.encoders['gender_encoder'].transform(self.interactions_df['gender'])

        known_countries = set(self.encoders['country_encoder'].classes_)
        self.interactions_df['country'] = self.interactions_df['country'].apply(lambda x: x if x in known_countries else 'unknown')
        self.interactions_df['country_idx'] = self.encoders['country_encoder'].transform(self.interactions_df['country'])

        # 3. Pre-procesamiento de secuencias
        self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        self.interactions_df.sort_values(by=['user_id', 'timestamp'], inplace=True)

        # Agrupar historial por usuario
        self.user_groups = self.interactions_df.groupby('user_id')['track_id'].apply(list).to_dict()

        # Guardar features tabulares en un array numpy para acceso rápido (float32)
        self.tabular_data = self.interactions_df[self.tabular_features].values.astype(np.float32)

        # Guardar track_ids y user_ids en arrays para acceso rápido
        self.track_ids = self.interactions_df['track_id'].values
        self.user_ids = self.interactions_df['user_id'].values
        self.user_genders = self.interactions_df['gender_idx'].values
        self.user_countries = self.interactions_df['country_idx'].values

        # Pre-calcular índices de secuencia
        self._precompute_sequence_indices()

    def get_encoders(self):
        """Retorna los encoders ajustados para usarlos en validación/test."""
        return self.encoders

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx):
        # 1. Identificar Usuario e Item Objetivo
        user_id = self.user_ids[idx]
        target_track_id = self.track_ids[idx]
        user_gender = self.user_genders[idx]
        user_country = self.user_countries[idx]

        # 2. Construir Secuencia Histórica (User Tower Input)
        # Obtenemos toda la historia del usuario
        full_history = self.user_groups[user_id]

        # Encontrar el índice del item actual en la historia del usuario
        # Como el DF está ordenado, podemos asumir que la posición en full_history corresponde
        # al orden temporal. Pero 'idx' es global.
        # Estrategia más robusta:
        # El dataframe está ordenado. Podemos calcular el índice relativo dentro del grupo.
        # O simplemente, dado que es secuencial, el item en 'idx' corresponde al item en la posición T.
        # La historia son los items 0...T-1.

        # Para simplificar y no buscar en listas:
        # Podríamos haber pre-calculado los índices de inicio de cada usuario.
        # Pero asumamos que 'full_history' tiene todos los items.
        # Necesitamos saber CUÁL de esos items es el actual para hacer el slice [:current_pos].
        # Esto puede ser lento si buscamos.
        # OPTIMIZACIÓN: Asumimos que el dataset se usa tal cual está ordenado.
        # Pero si usamos Shuffle en el DataLoader, perdemos el orden.
        # Solución: Pre-calcular la posición en la secuencia para cada fila es costoso en memoria?
        # No, es solo un entero.

        # Vamos a asumir una implementación "Casual" donde tomamos una ventana aleatoria o
        # los últimos N items vistos antes de este timestamp.
        # Dado que self.user_groups tiene la lista ordenada cronológicamente:
        # Necesitamos el índice de ESTA interacción en la lista del usuario.
        # Lo calcularemos al vuelo o lo pre-calculamos en __init__.
        # Para este ejemplo, buscaremos el índice (puede ser lento O(N_user)).
        # MEJORA: Pre-calcular 'history_indices' en __init__.

        # Por ahora, implementamos la búsqueda (safe but slow) o asumimos pre-calculo.
        # Vamos a hacer un "hack" eficiente:
        # Usamos el índice global 'idx' si garantizamos que no hay shuffle en el DF base,
        # pero el DataLoader hará shuffle de índices.
        # Lo correcto: Agregar una columna 'user_seq_idx' en __init__.

        # (Ver implementación en __init__ abajo, aquí asumimos que existe)
        seq_idx = self.interactions_df_seq_idx[idx] # Necesitamos agregar esto

        history_items = full_history[:seq_idx]
        # Truncar o Padding
        if len(history_items) > self.max_seq_len:
            history_items = history_items[-self.max_seq_len:]

        # Mapear a enteros
        history_ids = [self.item_id_mapper.get(tid, 0) for tid in history_items]

        # Padding
        seq_len = len(history_ids)
        pad_len = self.max_seq_len - seq_len
        history_ids = history_ids + [0] * pad_len # 0 as padding index

        # Máscara de atención (1 para real, 0 para padding)
        attention_mask = [1] * seq_len + [0] * pad_len

        # 3. Cargar Modalidades del Item Objetivo (Item Tower Input)

        # A. Tabular
        tabular_feats = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # B. Imagen (Album Cover)
        img_path = os.path.join(self.img_dir, f"{target_track_id}.jpg")
        # Fallback si no existe imagen (imagen negra)
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except Exception:
                image = torch.zeros((3, 224, 224)) # Tensor negro
        else:
            image = torch.zeros((3, 224, 224))

        # C. Audio (Mel Spectrogram)
        if self.audio_dir:
            audio_path = os.path.join(self.audio_dir, f"{target_track_id}.pt")

            if os.path.exists(audio_path):
                try:
                    audio = torch.load(audio_path)
                    # Asegurar dimensiones (C, H, W) -> (1, 128, 128)
                    if audio.dim() == 2:
                        audio = audio.unsqueeze(0)
                except Exception:
                    audio = torch.zeros((1, 128, 128))
            else:
                # Si no hay audio, retornar tensor de ceros con shape compatible
                audio = torch.zeros((1, 128, 128))
        else:
            audio = torch.tensor([]) # Empty

        # D. Texto (Lyrics Embeddings)
        # Inicializar máscara de texto (1 = presente, 0 = ausente)
        text_mask = 0
        if self.text_embeddings and target_track_id in self.text_embeddings:
            text_emb = self.text_embeddings[target_track_id]
            text_mask = 1
        else:
            text_emb = torch.zeros(768) # Dimensión base de DistilBERT

        return {
            'user_id': user_id, # String, cuidado con DataLoader default collate
            'user_gender': torch.tensor(user_gender, dtype=torch.long),
            'user_country': torch.tensor(user_country, dtype=torch.long),
            'history_ids': torch.tensor(history_ids, dtype=torch.long),
            'history_mask': torch.tensor(attention_mask, dtype=torch.long),
            'target_image': image,
            'target_audio': audio,
            'target_text': text_emb,
            'target_text_mask': torch.tensor(text_mask, dtype=torch.long), # Máscara explicita
            'target_tabular': tabular_feats,
            'target_id': self.item_id_mapper.get(target_track_id, 0) # Para validación/métricas
        }

    def _precompute_sequence_indices(self):
        """Helper para asignar índice secuencial a cada fila por usuario."""
        # Esto es crítico para saber qué parte de la historia corresponde a cada fila
        # sin buscar en listas.
        self.interactions_df['seq_idx'] = self.interactions_df.groupby('user_id').cumcount()
        self.interactions_df_seq_idx = self.interactions_df['seq_idx'].values
