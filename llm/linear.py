import math
import torch


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.empty(
            size=(out_features, in_features),
            device=device,
            dtype=dtype
        ))
        self.reset_parameters()

    def reset_parameters(self):
        # sigma^2 = 2 / (in_features + out_features)
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input.
        return x @ self.weight.T
