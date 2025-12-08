import torch
import torch.nn as nn
import os
import torchvision.models as models
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
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
    def __init__(self, model_name="microsoft/mdeberta-v3-base", embedding_dim=128, use_lora=True):
        super().__init__()
        
        # Cargar modelo base
        cache_dir = os.getenv('HF_HOME')
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        if use_lora:
            # Configuración LoRA
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=8,            # Rango de la matriz de adaptación
                lora_alpha=32,  # Factor de escala
                lora_dropout=0.1,
                target_modules=["query_proj", "value_proj"] # Módulos a adaptar en DeBERTa
            )
            self.transformer = get_peft_model(self.transformer, peft_config)
            self.transformer.print_trainable_parameters()
            
        # Proyección final
        # mDeBERTa-v3-base hidden size es 768
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # Forward pass del Transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean Pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        
        # Proyección
        return self.projection(sentence_embeddings)

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
                 text_model_name: str = "microsoft/mdeberta-v3-base",
                 text_dim: int = 128,
                 tabular_dim: int = 128,
                 use_lora: bool = True):
        """
        Item Tower que fusiona 4 modalidades: Audio, Visual, Texto, Tabular.
        """
        super().__init__()
        
        self.audio_encoder = AudioEncoder(embedding_dim=audio_dim)
        self.visual_encoder = VisualEncoder(embedding_dim=visual_dim)
        self.text_encoder = TextEncoder(model_name=text_model_name, embedding_dim=text_dim, use_lora=use_lora)
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
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                tabular: torch.Tensor):
        
        # 1. Encode Modalities
        audio_emb = self.audio_encoder(audio)
        visual_emb = self.visual_encoder(images)
        text_emb = self.text_encoder(input_ids, attention_mask)
        tabular_emb = self.tabular_encoder(tabular)
        
        # 2. Concatenate
        # Si alguna modalidad falta (ej. audio vacío), deberíamos manejarlo con máscaras o tensores cero.
        # Aquí asumimos que el Dataset entrega tensores válidos (aunque sean ceros).
        combined = torch.cat([audio_emb, visual_emb, text_emb, tabular_emb], dim=1)
        
        # 3. Fusion
        item_embedding = self.fusion_layer(combined)
        
        return item_embedding




