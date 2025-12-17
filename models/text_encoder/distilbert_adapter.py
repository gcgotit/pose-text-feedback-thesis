import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBERTTextEncoder(nn.Module):
    def __init__(self, pretrained_model='distilbert-base-uncased', output_dim=128):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, texts):
        """
        Args:
            texts (List[str]): lista di stringhe da cui estrarre i text embeddings.

        Returns:
            Tensor di shape (B, output_dim)
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}

        outputs = self.encoder(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_embedding)

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
