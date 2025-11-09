from jaxtyping import Bool, Float
import torch

from llm.functional import softmax
from llm.linear import Linear
from llm.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... queries d_k"],
    K: Float[torch.Tensor, "batch_size ... keys d_k"],
    V: Float[torch.Tensor, "batch_size ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, "batch_size ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    ret: Float[torch.Tensor, "... queries keys"] = Q @ K.transpose(-1, -2) / (Q.shape[-1]**.5)
    if mask is not None:
        # add -torch.inf to the masked positions
        ret = ret.masked_fill(~mask, -torch.inf)
    return softmax(ret, -1) @ V


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 100_00.,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.theta = theta

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(
            theta=self.theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=self.device
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        Q: torch.Tensor = self.q_proj(x)
        K: torch.Tensor = self.k_proj(x)
        V: torch.Tensor = self.v_proj(x)

        batch_size, seq_len, d_model = x.shape

        # Reshape for multi-head attention: (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device, dtype=torch.long)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(
                (seq_len, seq_len),
                device=self.device,
                dtype=torch.bool)
            )

        out = scaled_dot_product_attention(Q, K, V, mask)

        return self.output_proj(out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        ))
