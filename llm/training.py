import numpy as np
import torch
import numpy.typing as npt

from cs336_basics.tokenizer import Tokenizer
from llm.ckpt import save_checkpoint
from llm.optimizer import AdamW, cross_entropy_loss
from llm.transformer import LLM


vocab_size = 50256 + 1
context_length = 128
d_model = 512
d_ff = 1314
theta = 10_000
num_layers = 4
num_heads = 16
batch_size = 8
n_epoch = 100
device = "cuda"


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = len(dataset)
    m = context_length
    if n <= m:
        raise ValueError(f"Sequence length n={n} must be greater than context_length={m}")

    # 最大可采样起始位置是 n - m - 1 (inclusive)
    # numpy.randint(low, high) 取值范围是 [low, high)
    starts = np.random.randint(0, n - m, size=batch_size)  # shape (B,)

    # 构造索引矩阵：每一行是 start + [0,1,...,m-1]
    offsets = np.arange(m)  # shape (m,)
    idx = starts[:, None] + offsets[None, :]  # shape (B, m)

    # inputs: dataset[idx] -> shape (B, m)
    inputs_np = dataset[idx]

    # targets: shift by 1 -> indices idx + 1
    targets_np = dataset[idx + 1]

    # 转为 torch tensors 放到 device（long 类型）
    inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)

    return inputs, targets


def train():

    tokenizer = Tokenizer.from_files(
        vocab_filepath='tests/fixtures/gpt2_vocab.json',
        merges_filepath='tests/fixtures/gpt2_merges.txt',
        special_tokens=["<|endoftext|>"]
    )

    # vocabs = list(tokenizer.vocab.items())
    # merges = list(tokenizer.merges)
    # print(f"{vocabs[0]=}, {vocabs[1]=}, {vocabs[2]=}")
    # print(f"{merges[0]=}, {merges[1]=}, {merges[2]=}")

    # read from tests/fixtures/corpus.en and tokenize
    with open('tests/fixtures/corpus.en', 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    return

    np.save("tokens.npy", tokens)

    llm = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=theta,
        device=torch.device(device),
        dtype=torch.float32
    )

    optimizer = AdamW(params=llm.parameters())

    # Dummy training loop
    for epoch in range(n_epoch):
        inputs, outputs = get_batch(
            dataset=np.load("tokens.npy", mmap_mode='r'),
            batch_size=batch_size,
            context_length=context_length,
            device=device
        )
        optimizer.zero_grad()
        logits = llm(inputs)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), outputs.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.cpu().item()}")

        save_checkpoint(llm, optimizer, epoch + 1, f"checkpoint_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    train()
