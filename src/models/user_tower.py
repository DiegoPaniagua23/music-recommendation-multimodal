import torch
import torch.nn as nn

class SequentialUserEncoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int = 256, 
                 max_seq_len: int = 50, 
                 num_heads: int = 4, 
                 num_layers: int = 2, 
                 dropout: float = 0.1):
        """
        User Tower basada en SASRec (Self-Attentive Sequential Recommendation).
        Codifica la secuencia de interacciones pasadas en un vector de usuario.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Item Embeddings (ID-based)
        # Padding idx 0 is handled by setting padding_idx=0
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional Embeddings
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=embedding_dim * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN suele ser más estable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización de pesos (Xavier)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, history_ids: torch.Tensor, history_mask: torch.Tensor = None):
        """
        Args:
            history_ids: (Batch, Seq_Len) - IDs de items interactuados.
            history_mask: (Batch, Seq_Len) - 1 para válido, 0 para padding.
        Returns:
            user_embedding: (Batch, Embedding_Dim) - Representación del estado del usuario.
        """
        batch_size, seq_len = history_ids.shape
        
        # 1. Embeddings
        items_emb = self.item_embedding(history_ids) # (B, L, D)
        
        # Positions: [0, 1, 2, ..., L-1]
        positions = torch.arange(seq_len, device=history_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions) # (B, L, D)
        
        x = items_emb + pos_emb
        x = self.dropout(self.layer_norm(x))
        
        # 2. Máscaras
        # Padding Mask para Transformer (True = ignorar)
        # history_mask es 1 para válido, 0 para padding.
        # PyTorch Transformer espera:
        #   src_key_padding_mask: (Batch, Seq_Len) donde True significa ignorar.
        if history_mask is not None:
            src_key_padding_mask = (history_mask == 0)
        else:
            src_key_padding_mask = (history_ids == 0)
            
        # Causal Mask (Auto-regresiva)
        # Asegura que la posición i solo pueda atender a 0...i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=history_ids.device)
        
        # 3. Transformer
        # Output: (B, L, D)
        out = self.transformer_encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True
        )
        
        # 4. Gather last valid item embedding
        # Queremos el embedding correspondiente a la última interacción real.
        
        # Calcular longitudes reales
        if history_mask is not None:
            lengths = history_mask.sum(dim=1) - 1 # Índice es longitud - 1
        else:
            lengths = (history_ids != 0).sum(dim=1) - 1
            
        # Clamp para evitar -1 si la secuencia está vacía (no debería pasar si se filtra)
        lengths = lengths.clamp(min=0)
        
        # Gather: out[b, length[b], :]
        # (B, D)
        user_embedding = out[torch.arange(batch_size, device=history_ids.device), lengths]
        
        return user_embedding
