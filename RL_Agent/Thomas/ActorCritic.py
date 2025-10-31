from typing import Tuple
import torch as T
import torch.nn as nn

from RL_Agent.Thomas.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = T.device("cpu")


class ActorCritic(T.jit.ScriptModule):
    def __init__(
        self,
        n_heads: int = 8,
        n_gae_layers: int = 3,
        obs_dim: int = 11,  # number of features
        embed_dim: int = 128,  # number of nodes
        gae_ff_hidden: int = 512,
        tanh_clip: float = 10.0,
        device=CPU_DEVICE,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads

        # Shared encoder
        self.gae = GraphAttentionEncoder(
            n_heads=n_heads,
            n_layers=n_gae_layers,
            embed_dim=embed_dim,     # encoder output dim E
            obs_dim=obs_dim,      # per-node input dim
            feed_forward_hidden=gae_ff_hidden,
        )

        # Actor head: per-node 2-class logits
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # Critic head: graph (pooled) embedding -> scalar
        self.value_layers = nn.Sequential(
            nn.Linear(embed_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        self.to(device)

    @T.jit.script_method
    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        features: [B, N, obs_dim]
        returns:
          probs   : [B, N, 2]  (softmax over class dimension per node)
          entropy : [B]        (mean per-node entropy per sample)
          values  : [B]        (critic values; negative sign preserved)
        """
        # Expect GAEncoder to return (embeddings: [B,N,E])
        embeddings = self.gae(features)
        mean_embeddings = embeddings.mean(dim=1)
        # --- Actor ---
        logits = self.prob_head(embeddings)          # [B, N, 2]
        probs = T.softmax(logits, dim=-1)         # [B, N, 2]

        # --- Critic ---
        values = self.value_layers(mean_embeddings).squeeze(-1)          # [B]
        values = -values  # keep your original sign convention

        return probs, values
