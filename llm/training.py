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


def tokenize():
    tokenizer = Tokenizer.from_files(
        vocab_filepath='tests/fixtures/gpt2_vocab.json',
        merges_filepath='tests/fixtures/gpt2_merges.txt',
        special_tokens=["<|endoftext|>"]
    )
    tokenizer.use_gpt_mapping = True

    # vocabs = list(tokenizer.vocab.items())
    # merges = list(tokenizer.merges)
    # print(f"{vocabs[0]=}, {vocabs[1]=}, {vocabs[2]=}")
    # print(f"{merges[0]=}, {merges[1]=}, {merges[2]=}")

    # read from tests/fixtures/corpus.en and tokenize
    with open('tests/fixtures/tinystories_sample_5M.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    np.save("tokens_tinystories_sample_5M.npy", tokens)

def train():
    llm = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=theta,
        device=torch.device(device),
        dtype=torch.float32
    )

    optimizer = AdamW(params=llm.parameters())
    tokens = np.load("tokens_tinystories_sample_5M.npy", mmap_mode='r')

    # Dummy training loop
    for epoch in range(n_epoch):
        inputs, outputs = get_batch(
            dataset=tokens,
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

def test_fast():
    tokenizer = Tokenizer.from_files(
        vocab_filepath='tests/fixtures/gpt2_vocab.json',
        merges_filepath='tests/fixtures/gpt2_merges.txt',
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.use_gpt_mapping = True

    tests = [
        "hello world",
        # "这是中文",
        "some text that i'll pre-tokenize",
        # "🙂 emoji test",
        # "café naïve",
        "some rare char"
    ]

    for t in tests:
        ids = tokenizer.encode(t)
        print(ids)
        toks = tokenizer.decode(ids)
        # 检查每个 token 是否在 vocab 字典中（理论上都应该在）
        print("TEXT:", t)
        print("TOKENS:", toks)
        print("---")

def read_npy_basic(filepath):
    """读取npy文件并显示基本信息"""
    try:
        data = np.load(filepath)
        
        print("=== NPY 文件信息 ===")
        print(f"文件路径: {filepath}")
        print(f"数据形状: {data.shape}")
        print(f"数据维度: {data.ndim}")
        print(f"数据类型: {data.dtype}")
        print(f"数据大小: {data.size}")
        print(f"字节大小: {data.nbytes} bytes")
        print(f"内存布局: {data.flags}")
        
        # 显示前几个元素
        print(f"\n前5个元素: {data.flat[:5] if data.size > 5 else data.flat[:]}")
        
        # 显示统计信息
        if np.issubdtype(data.dtype, np.number):
            print(f"最小值: {data.min():.4f}")
            print(f"最大值: {data.max():.4f}")
            print(f"平均值: {data.mean():.4f}")
            print(f"标准差: {data.std():.4f}")
        
        return data
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


if __name__ == "__main__":
    # train()
    # test_fast()
    # read_npy_basic("tokens.npy")
    # prompt = input("Input a prompt:")
    # inference(prompt)
    train()
