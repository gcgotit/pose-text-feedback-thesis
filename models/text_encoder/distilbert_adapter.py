import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


class ContrastiveAdapter(nn.Module):
    """
    Adapter MLP per contrastive learning.
    Trasforma l'embedding CLS di DistilBERT in uno spazio più adatto
    per l'allineamento contrastivo con gli embedding delle pose.
    
    Architettura:
    - Linear down-projection (bottleneck)
    - GELU activation
    - Linear up-projection
    - Residual connection (se dimensioni compatibili)
    """
    def __init__(self, input_dim, bottleneck_dim=256, output_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Residual connection solo se input_dim == output_dim
        self.use_residual = (input_dim == output_dim)
        
        # Inizializzazione per adapter (near-identity all'inizio)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.up_proj(out)
        
        if self.use_residual:
            out = out + x
            
        out = self.layer_norm(out)
        return out


class DistilBERTTextEncoder(nn.Module):
    """
    Text encoder basato su DistilBERT con supporto per:
    - Freezing dei layer BERT per ridurre memoria e velocizzare il training
    - Adapter MLP trainabile per contrastive learning
    - Projection layer finale per allineare la dimensionalità con il pose encoder
    
    Args:
        pretrained_model: Nome del modello DistilBERT pretrained
        output_dim: Dimensione dell'embedding finale (default: 128)
        freeze_bert: Se True, congela tutti i parametri di DistilBERT (default: True)
        use_adapter: Se True, aggiunge un Adapter MLP trainabile tra BERT e projection (default: True)
        adapter_bottleneck: Dimensione del bottleneck dell'adapter (default: 256)
    """
    def __init__(
        self, 
        pretrained_model='distilbert-base-uncased', 
        output_dim=128,
        freeze_bert=True,
        use_adapter=True,
        adapter_bottleneck=256
    ):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        
        hidden_size = self.encoder.config.hidden_size  # 768 per distilbert-base
        
        # Freezing di DistilBERT
        self.freeze_bert = freeze_bert
        if freeze_bert:
            self._freeze_bert_parameters()
            print(f"🧊 DistilBERT congelato ({sum(1 for _ in self.encoder.parameters())} parametri)")
        
        # Adapter MLP (trainabile anche se BERT è congelato)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = ContrastiveAdapter(
                input_dim=hidden_size,
                bottleneck_dim=adapter_bottleneck,
                output_dim=hidden_size  # Mantiene la stessa dimensione per residual
            )
            print(f"🔧 Adapter attivo (bottleneck={adapter_bottleneck})")
        else:
            self.adapter = None
        
        # Projection layer finale (sempre trainabile)
        self.projection = nn.Linear(hidden_size, output_dim)
        
        # Log parametri trainabili
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Text Encoder: {trainable_params:,} trainabili / {total_params:,} totali")

    def _freeze_bert_parameters(self):
        """Congela tutti i parametri di DistilBERT."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        """Scongela tutti i parametri di DistilBERT (utile per fine-tuning)."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True
        self.freeze_bert = False
        print("🔥 DistilBERT scongelato")

    def forward(self, input_ids, attention_mask):
        # Passa attraverso DistilBERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Passa attraverso l'adapter (se attivo)
        if self.use_adapter and self.adapter is not None:
            cls_embedding = self.adapter(cls_embedding)
        
        # Projection finale
        projected = self.projection(cls_embedding)
        return projected


if __name__ == "__main__":
    # Test con configurazione di default (BERT congelato + adapter)
    print("=== Test DistilBERTTextEncoder ===\n")
    
    model = DistilBERTTextEncoder(
        freeze_bert=True,
        use_adapter=True,
        adapter_bottleneck=256
    )
    model.eval()
    
    # Verifica parametri trainabili
    print("\nParametri trainabili:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}: {param.numel():,}")
    
    # Test forward pass
    texts = [
        "Svend Press. Push the dumbbells straight ahead until your arms are fully extended.",
        "Push-ups. Maintain a straight body and keep elbows close."
    ]
    
    # Tokenizza
    encoded = model.tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
        print(f"\nText embedding shape: {embeddings.shape}")  # Expected: (2, 128)
