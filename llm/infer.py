from cs336_basics.tokenizer import Tokenizer
from llm.transformer import LLM
import torch

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

def inference(
    prompt: str,
    ckpt_epoch: int = 1000,
    gen_seq_len: int = 128
):
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
    llm.load_state_dict(torch.load(f"checkpoint_epoch_{ckpt_epoch}.pt")['model_state_dict'])

    tokenizer = Tokenizer.from_files(
        vocab_filepath='tests/fixtures/gpt2_vocab.json',
        merges_filepath='tests/fixtures/gpt2_merges.txt',
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.use_gpt_mapping = True

    input_ids = tokenizer.encode(prompt)

    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

        generated_ids = input_ids.copy()

        for _ in range(gen_seq_len):  # Generate 50 tokens
            logits = llm(input_tensor)
            # print(f"{logits=}")
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            generated_ids.append(next_token_id)
            # print(f"Generated ids: {generated_ids=}")
            input_tensor = torch.tensor([generated_ids[-context_length:]], device=device, dtype=torch.long)
            # print(input_tensor)

        generated_text = tokenizer.decode(generated_ids)
        print("Generated Text:")
        print(generated_text)


prompt_1 = "Once upon a time, there was a pretty girl named Lily. She loved to"


if __name__ == "__main__":
    prompt_text = "Once upon a time, a prince and a princess lived in a grand castle."
    inference(prompt_1, ckpt_epoch=20000, gen_seq_len=512)
