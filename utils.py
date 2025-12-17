import torch
import torch.nn.functional as F

def ntxent_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Calcola la Normalized Temperature-scaled Cross Entropy Loss (NTXentLoss) tra due insiemi di embedding.
    
    Args:
        embeddings1 (torch.Tensor): Tensore degli embedding del primo dominio (es. pose), 
                                    dimensioni (N, D) dove N è il numero di campioni del batch e D la dimensionalità.
        embeddings2 (torch.Tensor): Tensore degli embedding del secondo dominio (es. testo),
                                    dimensioni (N, D), corrispondenti uno-a-uno con embeddings1.
        temperature (float, opzionale): Parametro di temperatura τ per scalare le similitudini (default = 0.5).
        
    Returns:
        torch.Tensor: Un tensore scalare (0-dimension) con il valore della loss contrastiva NTXent.
    """
    # Verifica che il numero di esempi corrisponda
    if embeddings1.shape[0] != embeddings2.shape[0]:
        raise ValueError("I tensori embeddings1 ed embeddings2 devono avere lo stesso numero di campioni.")
    
    # Assicura che i tensori siano sullo stesso dispositivo (CPU o GPU)
    if embeddings1.device != embeddings2.device:
        embeddings2 = embeddings2.to(embeddings1.device)
    
    # Normalizza gli embedding lungo la dimensione delle caratteristiche (D) per ottenere vettori unitari
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
    
    # Calcola la matrice di similarità (dot product) tra tutti gli embedding
    # Risulterà una matrice N x N dove entry (i,j) = sim(embeddings1[i], embeddings2[j])
    similarity_matrix = embeddings1_norm @ embeddings2_norm.T  # prodotto matrice (N,D)x(D,N) -> (N,N)
    
    # Crea il target per il calcolo della cross-entropy: per ogni riga i, il "label corretto" è i (coppia positiva)
    device = similarity_matrix.device
    N = similarity_matrix.size(0)
    target = torch.arange(N, device=device)
    
    # Calcola la loss cross-entropia per le due direzioni:
    # 1. Considerando embeddings1 come anchor e classificando embeddings2
    loss_i = F.cross_entropy(similarity_matrix / temperature, target, reduction='mean')
    # 2. Considerando embeddings2 come anchor e classificando embeddings1 
    loss_j = F.cross_entropy(similarity_matrix.T / temperature, target, reduction='mean')
    
    # Media delle due loss (posizioni positive considerate da entrambe le prospettive)
    loss = 0.5 * (loss_i + loss_j)
    return loss
