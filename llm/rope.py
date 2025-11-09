from jaxtyping import Float
import torch

class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(max_seq_len, device=device).unsqueeze(1)
        k_indices = torch.arange(
            start=1,
            end=d_k // 2 + 1,
            device=device
        )
        freqs = 1. / (self.theta ** ((2 * k_indices - 2) / d_k))
        angles = pos * freqs.unsqueeze(0)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"]
    ) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos = self.cos[token_positions]  # broadcast 到 batch
        sin = self.sin[token_positions]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
