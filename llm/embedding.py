import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings: int Size of the vocabulary
        # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.empty(
            size=(num_embeddings, embedding_dim),
            device=device,
            dtype=torch.float32
        ), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        sigma = 1.
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        # The forward method should select the embedding vector
        # for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        # torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        return self.weight[token_ids]
