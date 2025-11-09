import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.full(
            size=(d_model,),
            fill_value=eps,
            device=device,
            dtype=torch.float32
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS Norm
        rms = torch.pow(x, 2).add(self.eps).mean(dim=-1, keepdim=True).sqrt()
        result = (x / rms) * self.weight
        return result.to(in_dtype)
