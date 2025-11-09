from jaxtyping import Float

from llm.linear import Linear
from .attention import MultiHeadSelfAttention
from .swiglu import SwiGLU
from .rms_norm import RMSNorm
from .embedding import Embedding
import torch


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len d_model"]
    ) -> Float[torch.Tensor, "batch_size seq_len d_model"]:
        out = x + self.attn(self.ln1(x))
        return out + self.ffn(self.ln2(out))


class LLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))

