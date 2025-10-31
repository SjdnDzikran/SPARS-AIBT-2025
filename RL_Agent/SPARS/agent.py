import torch.nn as nn


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, device, hidden_sizes=(128, 128)):
        super().__init__()
        # shared body
        self.device = device

        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)

        # separate heads
        self.policy_head = nn.Linear(last, act_dim)
        self.value_head = nn.Linear(last, 1)

        self.to(device)

    def forward(self, obs):
        x = self.body(obs)
        logits = self.policy_head(x)        # [B, act_dim]
        value = self.value_head(x).squeeze(-1)  # [B]
        return logits, value
