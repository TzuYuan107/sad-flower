import torch
import torch.nn as nn

class Maze2dNNDynamicsModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)  # Output is Δs (state difference)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        delta_s = self.net(x)
        return s + delta_s # Euler integration