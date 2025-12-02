import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # ResNet18 modificada para entrada de 1 canal (Mel Spectrogram)
        # No usamos pesos pre-entrenados de ImageNet aquí porque los espectrogramas 
        # son muy diferentes a fotos naturales.
        self.backbone = models.resnet18(weights=None)
        
        # Modificar la primera capa convolucional para aceptar 1 canal en lugar de 3
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Reemplazar la capa fully connected final
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

class VisualEncoder(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        # Usamos pesos pre-entrenados por defecto (ImageNet)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Reemplazar la capa fully connected final
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

class TextEncoder(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=128):
        super().__init__()
        # Proyección simple para embeddings de Transformers (DistilBERT, mDeBERTa, etc.)
        # mDeBERTa-v3-base tiene dim=768, igual que BERT base.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        return self.projection(x)

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        # MLP para features tabulares (numéricas + one-hot encoded)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class MultimodalItemEncoder(nn.Module):
    def __init__(self, 
                 tabular_input_dim: int,
                 embedding_dim: int = 256,
                 audio_dim: int = 128,
                 visual_dim: int = 128,
                 text_input_dim: int = 768, # Nuevo parámetro para flexibilidad (ej. 1024 para Large)
                 text_dim: int = 128,
                 tabular_dim: int = 128):
        """
        Item Tower que fusiona 4 modalidades: Audio, Visual, Texto, Tabular.
        """
        super().__init__()
        
        self.audio_encoder = AudioEncoder(embedding_dim=audio_dim)
        self.visual_encoder = VisualEncoder(embedding_dim=visual_dim)
        self.text_encoder = TextEncoder(input_dim=text_input_dim, embedding_dim=text_dim)
        self.tabular_encoder = TabularEncoder(input_dim=tabular_input_dim, embedding_dim=tabular_dim)
        
        # Capa de Fusión (Late Fusion)
        fusion_input_dim = audio_dim + visual_dim + text_dim + tabular_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim) # Normalización final para estabilidad en InfoNCE
        )
        
    def forward(self, 
                images: torch.Tensor, 
                audio: torch.Tensor, 
                text: torch.Tensor, 
                tabular: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None):
        """
        Args:
            images: (B, 3, 224, 224)
            audio: (B, 1, F, T)
            text: (B, 768)
            tabular: (B, tabular_input_dim)
            text_mask: (B,) or (B, 1) - 1 si hay texto, 0 si no.
        Returns:
            item_embedding: (B, embedding_dim)
        """
        vis_emb = self.visual_encoder(images)
        aud_emb = self.audio_encoder(audio)
        txt_emb = self.text_encoder(text)
        tab_emb = self.tabular_encoder(tabular)
        
        # Aplicar máscara de texto si está disponible
        # Esto asegura que si no hay texto, la contribución sea exactamente 0 (sin sesgo/bias residual)
        if text_mask is not None:
            if text_mask.dim() == 1:
                text_mask = text_mask.unsqueeze(1)
            txt_emb = txt_emb * text_mask
        
        # Concatenación
        concat = torch.cat([vis_emb, aud_emb, txt_emb, tab_emb], dim=1)
        
        # Fusión
        item_embedding = self.fusion_layer(concat)
        
        return item_embedding
