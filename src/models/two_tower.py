import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .item_tower import MultimodalItemEncoder
from .user_tower import SequentialUserEncoder

class TwoTowerModel(nn.Module):
    def __init__(self, 
                 # User Tower Args
                 vocab_size: int,
                 max_seq_len: int = 50,
                 user_embedding_dim: int = 256,
                 user_num_heads: int = 4,
                 user_num_layers: int = 2,
                 user_dropout: float = 0.1,
                 # Item Tower Args
                 tabular_input_dim: int = 0,
                 item_embedding_dim: int = 256,
                 audio_dim: int = 128,
                 visual_dim: int = 128,
                 text_input_dim: int = 768,
                 text_dim: int = 128,
                 tabular_dim: int = 128,
                 # Loss Args
                 temperature: float = 0.07):
        """
        Modelo Two-Tower Multimodal completo.
        Combina SequentialUserEncoder y MultimodalItemEncoder.
        """
        super().__init__()
        
        # Asegurar que las dimensiones de salida de ambas torres sean iguales
        assert user_embedding_dim == item_embedding_dim, \
            f"User dim ({user_embedding_dim}) must match Item dim ({item_embedding_dim})"
            
        self.temperature = temperature
        
        # 1. User Tower
        self.user_tower = SequentialUserEncoder(
            vocab_size=vocab_size,
            embedding_dim=user_embedding_dim,
            max_seq_len=max_seq_len,
            num_heads=user_num_heads,
            num_layers=user_num_layers,
            dropout=user_dropout
        )
        
        # 2. Item Tower
        self.item_tower = MultimodalItemEncoder(
            tabular_input_dim=tabular_input_dim,
            embedding_dim=item_embedding_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            tabular_dim=tabular_dim
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Args:
            batch: Diccionario del DataLoader con claves:
                   - history_ids, history_mask (User Input)
                   - target_image, target_audio, target_text, target_tabular (Item Input)
                   - target_text_mask (Opcional)
        Returns:
            loss: Escalar (InfoNCE Loss)
            logits: Matriz de similitud (Batch, Batch)
        """
        # 1. Forward User Tower
        user_emb = self.user_tower(
            history_ids=batch['history_ids'],
            history_mask=batch['history_mask']
        ) # (B, D)
        
        # 2. Forward Item Tower
        item_emb = self.item_tower(
            images=batch['target_image'],
            audio=batch['target_audio'],
            text=batch['target_text'],
            tabular=batch['target_tabular'],
            text_mask=batch.get('target_text_mask')
        ) # (B, D)
        
        # 3. Normalización (L2 Norm)
        # Crucial para el producto punto (Cosine Similarity)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        # 4. Cálculo de Similitud (Logits)
        # (B, D) @ (D, B) -> (B, B)
        # logits[i, j] es la similitud entre usuario i e item j
        logits = torch.matmul(user_emb, item_emb.t()) / self.temperature
        
        # 5. Cálculo de Loss (InfoNCE Simétrica / CLIP Loss)
        # Los positivos son la diagonal (usuario i con item i)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Loss User -> Item (¿Cuál es el item correcto para este usuario?)
        loss_u2i = F.cross_entropy(logits, labels)
        
        # Loss Item -> User (¿Cuál es el usuario correcto para este item?)
        # Transponemos logits para que las filas sean items y columnas usuarios
        loss_i2u = F.cross_entropy(logits.t(), labels)
        
        # Promedio simétrico
        loss = (loss_u2i + loss_i2u) / 2
        
        return loss, logits, user_emb, item_emb

    def get_user_embedding(self, history_ids, history_mask=None):
        """Helper para inferencia solo de usuario"""
        emb = self.user_tower(history_ids, history_mask)
        return F.normalize(emb, p=2, dim=1)

    def get_item_embedding(self, images, audio, text, tabular, text_mask=None):
        """Helper para inferencia solo de item (para indexar catálogo)"""
        emb = self.item_tower(images, audio, text, tabular, text_mask)
        return F.normalize(emb, p=2, dim=1)
