from __future__ import annotations
from collections.abc import Iterable
import json
from typing import Iterator, Self
import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import SBS_PAT, merge_tuple
from utils.bytes_utils import bytes_list_to_string, bytes_list_to_string_gpt_mapping, string_to_bytes_tuple, string_to_bytes_tuple_gpt_mapping


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        use_gpt_mapping: bool = False
    ):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        # See en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character for more information about the Unicode
        # replacement character.
        # the following parameters:
        self.vocab = vocab
        self.bytes_to_id = {
            v: k for k, v in self.vocab.items()
        }
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_token_bytes_tuple: list[tuple[bytes, ...]] = [
            tuple(bb for bb in s.encode('utf-8')) for s in self.special_tokens
        ]

        self.merge_ranks = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }

        self.use_gpt_mapping = False

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens.
        vocab: dict[int, bytes] = {}
        # now assume vocab_filepath is a json file mapping bytes -> id
        # convert the file's content to vocab: dict[int, bytes] 
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            byte_to_id = json.load(f)
            for byte_str, id_ in byte_to_id.items():
                vocab[id_] = byte_str.encode('utf-8')

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                first, second = line.split(' ')
                merges.append((first.encode('utf-8'), second.encode('utf-8')))
        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[tuple[bytes, ...]]:
        result: list[str] = []
        i = 0

        with tqdm(total=len(text), desc="Pre-tokenizing", unit="char") as pbar:
            while i < len(text):
                start_i = i  # 记录本轮开始位置

                # 1. special token 匹配
                matched_special = None
                for s in self.special_tokens:
                    if text.startswith(s, i):
                        matched_special = s
                        break

                if matched_special:
                    result.append(matched_special)
                    i += len(matched_special)
                else:
                    # 2. regex 匹配
                    m = re.match(SBS_PAT, text[i:])
                    if m:
                        result.append(m.group(0))
                        i += len(m.group(0))
                    else:
                        # 3. fallback：前进一个字符
                        i += 1

                # 更新 tqdm（按字符数）
                pbar.update(i - start_i)

        if not self.use_gpt_mapping:
            return [string_to_bytes_tuple(token) for token in result]
        else:
            return [string_to_bytes_tuple_gpt_mapping(token) for token in result]

    def __merge_single_token(
        self,
        pretokens: tuple[bytes]
    ) -> tuple[bytes]:
        if pretokens in self.special_token_bytes_tuple:
            return pretokens

        while 1:
            ranked: list[tuple[int, tuple[bytes, bytes]]] = []
            for i in range(len(pretokens) - 1):
                if (pretokens[i], pretokens[i + 1]) in self.merge_ranks:
                    merging_tuple = (pretokens[i], pretokens[i + 1])
                    # print(f"Find a matched merge {merging_tuple=}")
                    ranked.append((self.merge_ranks[merging_tuple], merging_tuple))

            if not ranked:
                break

            _, best_pair = min(ranked, key=lambda x: x[0])
            new_token = []
            i = 0
            while i < len(pretokens):
                if i < len(pretokens) - 1 and (pretokens[i], pretokens[i + 1]) == best_pair:
                    new_token.append(pretokens[i] + pretokens[i + 1])
                    i += 2
                else:
                    new_token.append(pretokens[i])
                    i += 1
            pretokens = new_token

        return pretokens

    def encode(self, text: str) -> list[int]:
        tokens = self.pre_tokenize(text)
        # print(f"{tokens=}")

        ret = []
        for i, token in enumerate(tokens):
            merged_tokens = self.__merge_single_token(token)
            for b in merged_tokens:
                ret.append(self.bytes_to_id.get(b, -1))
            print(f"encode progress: {i + 1}/{len(tokens)}")

        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-eﬀicient tokenization of large files that we cannot directly load into
        # memory.
        pass

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        tokens = [self.vocab[id_] for id_ in ids]
        if not self.use_gpt_mapping:
            return bytes_list_to_string(tokens)
        else:
            return bytes_list_to_string_gpt_mapping(tokens)

def test_simple_encoding():
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

    tokenizer = Tokenizer(vocab, merges)
    ret = tokenizer.encode("the cat ate")
    print(ret)


if __name__ == '__main__':
    test_simple_encoding()
