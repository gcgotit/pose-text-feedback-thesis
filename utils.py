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


'''
Grad Norm perchè:

I due encoder (pose vs. text) hanno:

complessità diversa (pochi parametri vs. tanti),

gradienti sbilanciati (hai già visto dai CSV),

dinamiche di apprendimento divergenti.

GradNorm ti permette di imparare automaticamente quanto peso assegnare a ciascuna NTXentLoss, per non far sì che uno domini sull'altro.
'''
class GradNormLossWrapper:
    def __init__(self, loss_fns, initial_losses, alpha=1.5, device='cpu'):
        """
        loss_fns: List of loss functions, one per task (es. [pose_loss_fn, text_loss_fn])
        initial_losses: Initial loss values per task, to compute relative training rate
        alpha: GradNorm balancing exponent (default: 1.5)
        """
        self.loss_fns = loss_fns
        self.alpha = alpha
        self.device = device

        self.n_tasks = len(loss_fns)
        self.task_weights = torch.nn.Parameter(torch.ones(self.n_tasks, device=device))
        self.initial_losses = torch.tensor(initial_losses, device=device)

    def compute_loss(self, model, shared_params, inputs, targets):
        """
        Calcola la loss totale pesata + regolarizzazione GradNorm
        shared_params: parametri condivisi da cui si backpropaga (es. pose encoder + text encoder)
        inputs: lista di input per ciascun task
        targets: lista di target per ciascun task
        """
        task_losses = []
        task_grads = []

        for i in range(self.n_tasks):
            loss = self.loss_fns[i](inputs[i], targets[i])
            task_losses.append(loss)

            # Filtra i parametri condivisi per includere solo quelli che richiedono gradienti.
            # alcuni dei parametri potrebbero essere congelati (cioè requires_grad=False, ad esempio i layer di DistilBERT freezati).
            trainable_params = [p for p in shared_params if p.requires_grad]
            grads = torch.autograd.grad(
                loss, trainable_params,
                retain_graph=True, create_graph=True,
                allow_unused=True  # 👈 fix!
            )

            # Filtro: rimuovi None (non coinvolti nella loss)
            grads = [g for g in grads if g is not None]
            if grads:
                grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
            else:
                grad_norm = torch.tensor(0.0, device=self.device)

            task_grads.append(grad_norm)

        avg_loss = sum([self.task_weights[i] * task_losses[i] for i in range(self.n_tasks)])

        # Compute relative inverse training rate
        with torch.no_grad():
            loss_ratios = torch.tensor([task_losses[i].item() / self.initial_losses[i] for i in range(self.n_tasks)], device=self.device)
            avg_ratio = loss_ratios.mean()
            inverse_training_rate = loss_ratios / avg_ratio

        # Compute target gradient norm
        mean_grad = torch.stack(task_grads).mean()
        target_grads = mean_grad * (inverse_training_rate ** self.alpha)

        # Compute GradNorm loss (L1 loss between target and actual grads)
        gradnorm_loss = F.l1_loss(torch.stack(task_grads), target_grads.detach())

        # Combine task losses + gradnorm
        total_loss = avg_loss + gradnorm_loss
        return total_loss, task_losses, gradnorm_loss

    def get_weights(self):
        return self.task_weights.data.cpu().tolist()
