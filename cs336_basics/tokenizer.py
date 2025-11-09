from collections.abc import Iterable
import json
from typing import Iterator, Self
import regex as re

from cs336_basics.pretokenization_example import SBS_PAT, merge_tuple
from utils.bytes_utils import bytes_list_to_string, string_to_bytes_tuple


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
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

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
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
        while i < len(text):
            # 检查是否以 special token 开头
            matched_special = None
            for s in self.special_tokens:
                if text.startswith(s, i):
                    matched_special = s
                    break
            if matched_special:
                result.append(matched_special)
                i += len(matched_special)
                pass
            else:
                # 否则用 regex 匹配下一个 token
                m = re.match(SBS_PAT, text[i:])
                if m:
                    result.append(m.group(0))
                    i += len(m.group(0))
                else:
                    # 如果 regex 没匹配上，就前进一个字符
                    i += 1
        return [string_to_bytes_tuple(token) for token in result]

    def __merge_single_token(
        self,
        token: tuple[bytes]
    ) -> tuple[bytes]:
        if token in self.special_token_bytes_tuple:
            return token
        for merge in self.merges:
            token = merge_tuple(token, merge)
        return token

    def encode(self, text: str) -> list[int]:
        tokens = self.pre_tokenize(text)

        ret = []
        for token in tokens:
            merged_token = self.__merge_single_token(token)
            for b in merged_token:
                ret.append(self.bytes_to_id.get(b, -1))

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
        return bytes_list_to_string(tokens)

def test_simple_encoding():
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

    tokenizer = Tokenizer(vocab, merges)
    ret = tokenizer.encode("the cat ate")
    print(ret)


if __name__ == '__main__':
    test_simple_encoding()
