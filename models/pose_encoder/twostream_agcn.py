import torch
import torch.nn as nn
import torch.nn.functional as F

# Definizione del grafo scheletrico NTU-RGB+D
NTU_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

# Classe che costruisce il grafo e genera la matrice di adiacenza
class Graph:
    def __init__(self, num_nodes=25, edges=NTU_CONNECTIONS):
        self.num_nodes = num_nodes
        self.edges = edges
        self.A = self.build_adjacency_matrix()

    def build_adjacency_matrix(self):
        A = torch.eye(self.num_nodes) # matrice identità (25x25)
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1  # simmetrico
        D = torch.diag(A.sum(1)**-0.5)
        A_norm = D @ A @ D
        return A_norm

# Blocco spazio-temporale: GCN + TCN
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=9, stride=1):
        super().__init__()
        # Registra A come buffer per assicurarsi che venga spostato su GPU con .to(device)
        self.register_buffer('A', A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C, T, V)
        x = self.gcn(torch.einsum('vu, nctu -> nctv', self.A, x))
        x = self.bn(self.tcn(x))
        return F.relu(x)

# Architettura completa 2S-AGCN
class TwoStreamAGCN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, num_nodes=25):
        super().__init__()
        self.graph = Graph(num_nodes)
        A = self.graph.A

        # GCN blocks for position stream
        self.pos_gcn1 = STGCNBlock(input_dim, hidden_dim, A)
        self.pos_gcn2 = STGCNBlock(hidden_dim, hidden_dim, A)
        self.pos_gcn3 = STGCNBlock(hidden_dim, hidden_dim, A)

        # GCN blocks for velocity stream
        self.vel_gcn1 = STGCNBlock(input_dim, hidden_dim, A)
        self.vel_gcn2 = STGCNBlock(hidden_dim, hidden_dim, A)
        self.vel_gcn3 = STGCNBlock(hidden_dim, hidden_dim, A)

        # Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, V, C)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, T, V)
        x_pos = x
        x_vel = x[:, :, 1:, :] - x[:, :, :-1, :]
        x_vel = F.pad(x_vel, (0, 0, 0, 1))

        # Position stream
        x_pos = self.pos_gcn1(x_pos)
        x_pos = self.pos_gcn2(x_pos)
        x_pos = self.pos_gcn3(x_pos)

        # Velocity stream
        x_vel = self.vel_gcn1(x_vel)
        x_vel = self.vel_gcn2(x_vel)
        x_vel = self.vel_gcn3(x_vel)

        # Fuse
        x = x_pos + x_vel  # (B, C, T, V)
        x = x.mean(-1)     # average over joints -> (B, C, T)
        x = self.pool(x)   # (B, C, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        return self.fc(x)

# ==========================
# Script di test e conteggio parametri
# ==========================
if __name__ == "__main__":
    model = TwoStreamAGCN()
    dummy_input = torch.randn(2, 100, 25, 3)  # (batch_size=2, T=100, 25 joint, 3D)
    out = model(dummy_input)
    print("Output shape:", out.shape)

    # Conteggio parametri
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Numero totale di parametri: {total_params:,}")

    # Salvataggio opzionale
    with open("agcn_parametri.txt", "w") as f:
        f.write(f"Numero totale di parametri: {total_params:,}\n")

    print("✅ Test completato e parametri salvati su agcn_parametri.txt")
