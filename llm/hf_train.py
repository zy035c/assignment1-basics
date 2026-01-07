from __future__ import annotations
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer
from datasets import load_dataset
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
device = "cuda"

SMALL_LLM_CONFIG = dict(
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
)

MEDIUM_LLM_CONFIG = dict(
    d_model=1024,
    d_ff=4096,
    num_layers=24,
    num_heads=16,
)

LARGE_LLM_CONFIG = dict(
    d_model=2048,
    d_ff=5120,
    num_layers=36,
    num_heads=20,
)

XL_LLM_CONFIG = dict(
    d_model=1600,
    d_ff=6400,
    num_layers=48,
    num_heads=25,
)

TWOdot7B_LLM_CONFIG = dict(
    d_model=2560,
    d_ff=10240,
    num_layers=32,
    num_heads=32,
)

def download_hf_text_dataset(
    hf_dataset: str,
    split: str = "train",
    output_dir: str = "./hf_corpus",
    text_field: str = "text",
    max_samples: int | None = None,
):
    """
    下载 Hugging Face 文本数据集，并保存为本地 jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{hf_dataset.replace('/', '_')}_{split}.jsonl")

    dataset = load_dataset(hf_dataset, split=split)  # 非 streaming

    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, desc="Downloading HF dataset")):
            if max_samples and i >= max_samples:
                break

            text = example.get(text_field, "")
            if not text or not text.strip():
                continue

            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved dataset to: {out_path}")
    return out_path

def iterate_local_text_dataset(path: str):
    """
    从本地 jsonl 逐条读取文本
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text and text.strip():
                    yield text
            except Exception:
                continue

def pretrain_from_local_corpus(
    corpus_path: str,
    max_n_epoch=20000,
    batch_size=8,
    context_length=1024,
    device="cuda",
    tokenizer_vocab="tests/fixtures/gpt2_vocab.json",
    tokenizer_merges="tests/fixtures/gpt2_merges.txt",
    save_ckpt_every=500,
    early_stop_streak=5,
    config=SMALL_LLM_CONFIG,
):
    # --- 1. 初始化 tokenizer ---
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_vocab,
        merges_filepath=tokenizer_merges,
        special_tokens=["<|endoftext|>"]
    )
    tokenizer.use_gpt_mapping = True

    # --- 3. 初始化模型 ---
    llm = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=theta,
        device=torch.device(device),
        dtype=torch.float32
    )
    optimizer = AdamW(params=llm.parameters())

    losses = []
    last_loss = float('inf')
    loss_increase_streak = 0

    # --- 4. 训练循环 ---
    epoch = 0
    token_buffer = []

    for text in tqdm(iterate_local_text_dataset(corpus_path), desc="Training from local corpus"):
        try:
            tokens = tokenizer.encode(text).ids
        except Exception as e:
            print(f"Skipping segment due to tokenize error: {e}")
            continue

        if not tokens:
            continue

        token_buffer.extend(tokens)

        # 当 token_buffer 足够一个 batch 时进行训练
        while len(token_buffer) >= batch_size * context_length:
            # 构建 batch
            inputs = []
            outputs = []
            for _ in range(batch_size):
                chunk = token_buffer[:context_length + 1]
                token_buffer = token_buffer[context_length + 1:]

                inputs.append(chunk[:-1])
                outputs.append(chunk[1:])

            inputs = torch.tensor(inputs, dtype=torch.long, device=device)
            outputs = torch.tensor(outputs, dtype=torch.long, device=device)

            # 前向 + 反向 + 更新
            optimizer.zero_grad()
            logits = llm(inputs)
            loss = cross_entropy_loss(logits.view(-1, vocab_size), outputs.view(-1))
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().item()
            losses.append(current_loss)

            # 早停判断
            if current_loss > last_loss:
                loss_increase_streak += 1
            else:
                loss_increase_streak = 0

            last_loss = current_loss
            epoch += 1

            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.4f}, Tokens in buffer: {len(token_buffer)}")

            # 保存 checkpoint
            if epoch % save_ckpt_every == 0:
                save_checkpoint(llm, optimizer, epoch, f"checkpoint_epoch_{epoch}.pt")

            if loss_increase_streak >= early_stop_streak:
                print("Early stopping triggered.")
                save_checkpoint(llm, optimizer, epoch, f"checkpoint_epoch_{epoch}_final.pt")
                np.save("training_losses.npy", np.array(losses))
                return

    # --- 5. 保存最终结果 ---
    save_checkpoint(llm, optimizer, epoch, f"checkpoint_epoch_{epoch}_final.pt")
    np.save("training_losses.npy", np.array(losses))
    print("Training finished.")

if __name__ == "__main__":
    # pretrain_streaming()
    corpus_path = download_hf_text_dataset(
        hf_dataset="segyges/OpenWebText2",
        max_samples=5_000_000,
        output_dir="./hf_corpus"
    )
    # print(f"Corpus downloaded to: {corpus_path}")
