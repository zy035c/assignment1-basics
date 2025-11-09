from collections import Counter, defaultdict
import os
import regex as re
from pathlib import Path
from typing import BinaryIO

from utils.bytes_utils import string_to_bytes_list, string_to_bytes_tuple

current_file_path = Path(__file__).resolve()
proj_root_path = current_file_path.parent.parent

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(chunk: str, special_tokens: list[str]) -> list[str]:
    result = []
    i = 0
    while i < len(chunk):
        # 检查是否以 special token 开头
        matched_special = None
        for s in special_tokens:
            if chunk.startswith(s, i):
                matched_special = s
                break
        if matched_special:
            # result.append(matched_special)
            i += len(matched_special)
            pass
        else:
            # 否则用 regex 匹配下一个 token
            m = re.match(SBS_PAT, chunk[i:])
            if m:
                result.append(m.group(0))
                i += len(m.group(0))
            else:
                # 如果 regex 没匹配上，就前进一个字符
                i += 1
    return result

SBS_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_current_vocab(freqs: dict[tuple[bytes, ...], int]) -> int:
    return len(freqs.keys())

def merge_tuple(token: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_token = ()
    i = 0

    while i < len(token):
        if i == len(token) - 1:
            new_token = (*new_token, token[i])
            break

        b1, b2 = token[i], token[i + 1]

        if (b1, b2) == merge:
            new_token = (*new_token, b1 + b2)
            i += 2
        else:
            new_token = (*new_token, b1)
            i += 1

    if new_token != token:
        token = new_token
    return token

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    freqs: dict[tuple[bytes, ...], int] = defaultdict(lambda: 0)

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            tokens = pre_tokenize(chunk, special_tokens)
            for token in tokens:
                freqs[string_to_bytes_tuple(token)] += 1

    next_id = 0
    for s in special_tokens:
        vocab[next_id] = s.encode('utf-8')
        next_id += 1

    final_merges: list[tuple[bytes, bytes]] = []

    while count_current_vocab(freqs) > vocab_size - len(special_tokens):
        merges: Counter[tuple[bytes, bytes]] = Counter()

        for token, freq in freqs.items():
            for i in range(len(token) - 1):
                merges[(token[i], token[i + 1])] += freq

        # print(f"{len(freqs)=}\n")
        # print(f"{merges=}\n")
        pair, _ = merges.most_common(1)[0]

        print(f"\nMost common pair: {pair[0] + pair[1]}\n")
        # vocab[next_id] = pair[0] + pair[1]
        # next_id += 1
        current_merge = (pair[0], pair[1])
        final_merges.append(current_merge)

        # update freqs
        key_to_remove = set()
        kv_to_add = set()

        for token, freq in freqs.items():
            new_token = merge_tuple(token, current_merge)

            if new_token != token:
                key_to_remove.add(token)
                kv_to_add.add((new_token, freq))

        for k in key_to_remove:
            del freqs[k]

        for k, v in kv_to_add:
            freqs[k] += v

    # print(f"Vocabulary size: {vocab}")
    # print(vocab)

    for token in freqs.keys():
        vocab[next_id] = token
        next_id += 1

    return vocab, final_merges

if __name__ == '__main__':
    vocab, merges = train_bpe(
        input_path=proj_root_path / 'tests/fixtures/tinystories_sample_5M.txt',
        vocab_size=256,
        special_tokens=["<|endoftext|>"]
    )

    print(vocab)
    print(merges)
