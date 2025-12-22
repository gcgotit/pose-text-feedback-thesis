"""
twostream_stgcn_plus.py
-----------------------
Two-Stream ST-GCN+ Encoder per il dual encoder pose↔testo.

Architettura:
- Due stream separati: pose originali e velocità (differenze temporali)
- 4 blocchi ST-GCN+ per stream con canali progressivi (3→64→128→256→256)
- Residual connections, Batch Normalization, Dropout opzionale
- Fusione via concatenazione + MLP
- Output embedding 128D per loss contrastiva

Input: (B, T, V, C) dove B=batch, T=frames, V=25 joints, C=3 coords
Output: (B, 128) embedding normalizzato L2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GRAFO SCHELETRICO NTU-RGB+D (25 joints)
# ============================================================================

NTU_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),                          # Spina dorsale
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),         # Braccio sinistro
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),    # Braccio destro
    (0, 12), (12, 13), (13, 14), (14, 15),                     # Gamba sinistra
    (0, 16), (16, 17), (17, 18), (18, 19),                     # Gamba destra
]


class Graph:
    """
    Costruisce il grafo scheletrico e genera la matrice di adiacenza normalizzata.
    
    La normalizzazione simmetrica D^(-1/2) @ A @ D^(-1/2) garantisce stabilità
    nella propagazione del messaggio GCN.
    """
    def __init__(self, num_nodes=25, edges=NTU_CONNECTIONS):
        self.num_nodes = num_nodes
        self.edges = edges
        self.A = self.build_adjacency_matrix()

    def build_adjacency_matrix(self):
        """Costruisce matrice di adiacenza normalizzata simmetricamente."""
        A = torch.eye(self.num_nodes)  # Include self-loops
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1  # Grafo non direzionato
        
        # Normalizzazione simmetrica: D^(-1/2) @ A @ D^(-1/2)
        D = torch.diag(A.sum(1) ** -0.5)
        A_norm = D @ A @ D
        return A_norm


# ============================================================================
# BLOCCHI MODULARI
# ============================================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer.
    
    Applica convoluzione spaziale usando la matrice di adiacenza A.
    Formula: X' = A @ X @ W dove W è appreso dalla Conv2d.
    """
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # x: (B, C, T, V)
        # Aggregazione spaziale: A @ X (propagazione del messaggio sui nodi)
        x = torch.einsum('vu, nctu -> nctv', self.A, x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class TCNLayer(nn.Module):
    """
    Temporal Convolutional Layer.
    
    Applica convoluzione temporale con kernel dilazionato opzionale.
    Usa padding 'same' per mantenere la dimensione temporale.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        # x: (B, C, T, V)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class STGCNPlusBlock(nn.Module):
    """
    Blocco ST-GCN+ con residual connection.
    
    Struttura:
    1. GCN Layer (aggregazione spaziale)
    2. ReLU
    3. TCN Layer (aggregazione temporale)
    4. Residual connection (con proiezione 1x1 se necessario)
    5. ReLU finale
    
    Args:
        in_channels: canali in input
        out_channels: canali in output
        A: matrice di adiacenza normalizzata
        kernel_size: dimensione kernel temporale (default 9)
        stride: stride temporale (default 1)
        dropout: probabilità dropout (default 0.0)
        residual: se True, usa residual connection (default True)
    """
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1, dropout=0.0, residual=True):
        super().__init__()
        
        # Ramo principale: GCN → ReLU → TCN
        self.gcn = GCNLayer(in_channels, out_channels, A)
        self.tcn = TCNLayer(out_channels, out_channels, kernel_size, stride, dropout=dropout)
        
        # Residual connection
        self.use_residual = residual
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            # Proiezione 1x1 per matchare dimensioni
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, C, T, V)
        res = self.residual(x)
        
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        
        x = x + res
        x = self.relu(x)
        
        return x


# ============================================================================
# TWO-STREAM ST-GCN+ ENCODER
# ============================================================================

class TwoStreamSTGCNPlusEncoder(nn.Module):
    """
    Two-Stream ST-GCN+ Encoder per pose 3D.
    
    Architettura:
    - Stream 1 (Position): elabora le pose originali
    - Stream 2 (Velocity): elabora le differenze temporali (velocità)
    - Ogni stream ha 4 blocchi ST-GCN+ con canali progressivi
    - I due stream vengono concatenati e fusi via MLP
    - Output finale: embedding 128D normalizzato L2
    
    Args:
        input_dim: dimensione input per joint (default 3 per x,y,z)
        hidden_channels: lista canali hidden (default [64, 128, 256, 256])
        output_dim: dimensione embedding finale (default 128)
        num_nodes: numero joints (default 25)
        dropout: probabilità dropout nei blocchi (default 0.1)
        fusion_dropout: dropout nel MLP di fusione (default 0.3)
    """
    def __init__(
        self,
        input_dim=3,
        hidden_channels=None,
        output_dim=128,
        num_nodes=25,
        dropout=0.1,
        fusion_dropout=0.3
    ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 256]  # Canali progressivi
        
        self.output_dim = output_dim
        
        # Costruisci grafo scheletrico
        self.graph = Graph(num_nodes)
        A = self.graph.A
        
        # ==================== STREAM 1: POSITION ====================
        self.pos_blocks = nn.ModuleList()
        
        # Primo blocco: input_dim → hidden_channels[0]
        self.pos_blocks.append(
            STGCNPlusBlock(input_dim, hidden_channels[0], A, dropout=dropout, residual=False)
        )
        
        # Blocchi successivi con canali progressivi
        for i in range(len(hidden_channels) - 1):
            self.pos_blocks.append(
                STGCNPlusBlock(hidden_channels[i], hidden_channels[i+1], A, dropout=dropout)
            )
        
        # ==================== STREAM 2: VELOCITY ====================
        self.vel_blocks = nn.ModuleList()
        
        # Primo blocco: input_dim → hidden_channels[0]
        self.vel_blocks.append(
            STGCNPlusBlock(input_dim, hidden_channels[0], A, dropout=dropout, residual=False)
        )
        
        # Blocchi successivi con canali progressivi
        for i in range(len(hidden_channels) - 1):
            self.vel_blocks.append(
                STGCNPlusBlock(hidden_channels[i], hidden_channels[i+1], A, dropout=dropout)
            )
        
        # ==================== FUSION MLP ====================
        # Concatenazione dei due stream: hidden_channels[-1] * 2 → 256
        fusion_in_dim = hidden_channels[-1] * 2  # 256 * 2 = 512
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        
        # ==================== PROJECTION HEAD ====================
        # Proiezione finale: 256 → output_dim (128)
        self.projection = nn.Linear(256, output_dim)
        
        # Inizializzazione pesi
        self._init_weights()
    
    def _init_weights(self):
        """Inizializza i pesi delle layer lineari con Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _compute_velocity(self, x):
        """
        Calcola le differenze temporali (velocità).
        
        Args:
            x: tensor (B, C, T, V)
        
        Returns:
            velocity: tensor (B, C, T, V) con padding per mantenere shape
        """
        # Differenza temporale: frame[t+1] - frame[t]
        vel = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # Padding: ripeti l'ultimo frame di velocità per mantenere T
        vel = F.pad(vel, (0, 0, 0, 1), mode='replicate')
        
        return vel
    
    def forward(self, x, normalize=True):
        """
        Forward pass del Two-Stream ST-GCN+ Encoder.
        
        Args:
            x: tensor di pose (B, T, V, C) dove:
               - B = batch size
               - T = numero frames
               - V = numero joints (25)
               - C = coordinate (3)
            normalize: se True, normalizza L2 l'output (default True)
        
        Returns:
            embedding: tensor (B, output_dim) pronto per loss contrastiva
        """
        # Permuta input: (B, T, V, C) → (B, C, T, V)
        x = x.permute(0, 3, 1, 2)
        
        # Prepara i due stream
        x_pos = x                           # Pose originali
        x_vel = self._compute_velocity(x)   # Velocità (differenze temporali)
        
        # ==================== STREAM 1: POSITION ====================
        for block in self.pos_blocks:
            x_pos = block(x_pos)
        # x_pos: (B, 256, T, V)
        
        # ==================== STREAM 2: VELOCITY ====================
        for block in self.vel_blocks:
            x_vel = block(x_vel)
        # x_vel: (B, 256, T, V)
        
        # ==================== GLOBAL AVERAGE POOLING ====================
        # Pool su T e V separatamente
        # (B, C, T, V) → (B, C)
        x_pos = x_pos.mean(dim=[2, 3])  # Media su tempo e joints
        x_vel = x_vel.mean(dim=[2, 3])  # Media su tempo e joints
        
        # ==================== FUSION ====================
        # Concatenazione: (B, 256) + (B, 256) → (B, 512)
        x_fused = torch.cat([x_pos, x_vel], dim=1)
        
        # MLP di fusione: (B, 512) → (B, 256)
        x_fused = self.fusion_mlp(x_fused)
        
        # ==================== PROJECTION ====================
        # Proiezione finale: (B, 256) → (B, 128)
        embedding = self.projection(x_fused)
        
        # Normalizzazione L2 (per loss contrastiva)
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


# ============================================================================
# SCRIPT DI TEST E CONTEGGIO PARAMETRI
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: TwoStreamSTGCNPlusEncoder")
    print("=" * 60)
    
    # Crea modello
    model = TwoStreamSTGCNPlusEncoder(
        input_dim=3,
        hidden_channels=[64, 128, 256, 256],
        output_dim=128,
        num_nodes=25,
        dropout=0.1,
        fusion_dropout=0.3
    )
    
    # Dummy input: batch di 4 sequenze da 100 frame, 25 joint, 3 coordinate
    batch_size = 4
    T = 100
    V = 25
    C = 3
    
    dummy_input = torch.randn(batch_size, T, V, C)
    print(f"\n📥 Input shape: {dummy_input.shape}")
    print(f"   (B={batch_size}, T={T}, V={V}, C={C})")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n📤 Output shape: {output.shape}")
    print(f"   Embedding dimension: {output.shape[1]}")
    
    # Verifica normalizzazione L2
    norms = output.norm(p=2, dim=1)
    print(f"\n📐 L2 norms (should be ~1.0): {norms}")
    
    # Conteggio parametri
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pos_params = sum(p.numel() for p in model.pos_blocks.parameters() if p.requires_grad)
    vel_params = sum(p.numel() for p in model.vel_blocks.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.fusion_mlp.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in model.projection.parameters() if p.requires_grad)
    
    print(f"\n📊 Parametri:")
    print(f"   Position stream: {pos_params:,}")
    print(f"   Velocity stream: {vel_params:,}")
    print(f"   Fusion MLP:      {fusion_params:,}")
    print(f"   Projection head: {proj_params:,}")
    print(f"   ─────────────────────────")
    print(f"   TOTALE:          {total_params:,}")
    
    # Test su GPU (se disponibile)
    if torch.cuda.is_available():
        print(f"\n🖥️  Test GPU...")
        device = torch.device("cuda")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            output_gpu = model(dummy_input)
        
        print(f"   Output shape (GPU): {output_gpu.shape}")
        print(f"   Device: {output_gpu.device}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        cached = torch.cuda.memory_reserved(device) / 1024**2
        print(f"   Memory allocated: {allocated:.1f} MB")
        print(f"   Memory cached:    {cached:.1f} MB")
    
    print(f"\n✅ Test completato con successo!")
    print("=" * 60)

