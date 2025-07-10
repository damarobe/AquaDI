import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceHasher(nn.Module):
    def __init__(self, input_dim, hash_bits=32, use_gumbel=False, temperature=0.5):
        super(VoiceHasher, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, hash_bits)
        self.use_gumbel = use_gumbel
        self.temperature = temperature

    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        if self.use_gumbel:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(z) + 1e-10) + 1e-10)
            z = F.sigmoid((z + gumbel_noise) / self.temperature)
        else:
            z = torch.sigmoid(z)
        return torch.round(z)  # Binary hash code
