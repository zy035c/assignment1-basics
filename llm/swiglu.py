from jaxtyping import Bool, Float, Int
import torch
from .linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # glu = torch.nn.functional.sigmoid(x @ self.w1.T) * (x @ self.w2)
        x1: Float[torch.Tensor, "..."] = self.w1(x)
        silu = x1 * torch.nn.functional.sigmoid(x1)
        x2: Float[torch.Tensor, "... d_ff"] = silu * self.w3(x)
        return self.w2(x2)
