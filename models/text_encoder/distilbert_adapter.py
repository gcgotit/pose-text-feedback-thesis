import torch
import torch.nn as nn
from transformers import DistilBertModel

class AdapterLayer(nn.Module):
    """Adapter con collo di bottiglia: Linear -> ReLU -> Linear"""
    def __init__(self, hidden_size, bottleneck_dim=32):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x)))

class DistilBERTTextEncoder(nn.Module):
    def __init__(self, pretrained_model='distilbert-base-uncased', output_dim=128, freeze_layers=True, use_adapter=True):        
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)
        self.use_adapter = use_adapter

        if self.use_adapter:
            # Inserisci l'adapter solo nell'ultimo blocco
            hidden_size = self.encoder.config.hidden_size
            self.adapter = AdapterLayer(hidden_size)
        else:
            self.adapter = None

        if freeze_layers:
            self._freeze_distilbert_layers()

    def _freeze_distilbert_layers(self):
        # Congela tutti i parametri
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Scongela solo l'ultimo layer transformer (layer.5)
        for param in self.encoder.transformer.layer[-1].parameters():
            param.requires_grad = True

        print("[DistilBERTTextEncoder] Frozen all DistilBERT layers except last transformer layer, adapter and projection.")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        projected = self.projection(cls_embedding)
        return projected

    def encode_texts(self, texts, tokenizer, max_length=128):
        """Solo per uso in valutazione o test: tokenizza e calcola embedding"""
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            return self(inputs['input_ids'], inputs['attention_mask'])

def print_grad_status(model):
    print("\n" + "="*80)
    print("GRADIENT STATUS CHECK")
    print("="*80)

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            status = "✓ TRAINABLE"
        else:
            frozen_params += param.numel()
            status = "✗ FROZEN"

        if any(k in name for k in ['layer.5', 'projection', 'encoder.transformer']):
            print(f"{name:60} {status}")

    print("\n" + "="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print("="*80 + "\n")

    return trainable_params, total_params
