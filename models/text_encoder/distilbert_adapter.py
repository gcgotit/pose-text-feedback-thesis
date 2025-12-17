import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBERTTextEncoder(nn.Module):
    def __init__(self, pretrained_model='distilbert-base-uncased', output_dim=128):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
        projected = self.projection(cls_embedding)
        return projected

if __name__ == "__main__":
    model = DistilBERTTextEncoder()
    model.eval()
    texts = [
        "Svend Press. Push the dumbbells straight ahead until your arms are fully extended.",
        "Push-ups. Maintain a straight body and keep elbows close."
    ]
    with torch.no_grad():
        embeddings = model(texts)
        print("Text embedding shape:", embeddings.shape)  # Expected: (2, 128)
