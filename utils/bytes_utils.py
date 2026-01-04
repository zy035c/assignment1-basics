from __future__ import annotations

def string_to_bytes_list(string: str) -> list[bytes]:
    return [bytes([bb]) for bb in string.encode('utf-8')]

def string_to_bytes_tuple(string: str) -> tuple[bytes, ...]:
    return tuple(bytes([bb]) for bb in string.encode('utf-8'))

def string_to_bytes_tuple_gpt_mapping(string: str) -> tuple[bytes, ...]:
    res = ()
    for bb in string.encode('utf-8'):
        # print(f"{bb=}")
        mapped = BYTE_TO_UNICODE.get(bb, None)
        if mapped is not None:
            res = (*res, mapped.encode('utf-8'))
        else:
            res = (*res, bytes([bb]))
    return res


def bytes_tuple_to_string(bytes_tuple: tuple[bytes, ...]) -> str:
    return b"".join(bytes_tuple).decode()

def bytes_list_to_string(bytes_list: list[bytes]) -> str:
    return b"".join(bytes_list).decode()

def bytes_list_to_string_gpt_mapping(bytes_list: list[bytes]) -> str:
    decoded_str = b"".join(bytes_list).decode()
    ret = ""
    for char in decoded_str:
        char_ok = UNICODE_TO_BYTE.get(char, None)
        if char_ok is not None:
            ret += (bytes([char_ok])).decode('utf-8')
        else:
            ret += char
    return ret

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

if __name__ == "__main__":
    target = "Yes, indeed. It is called Lothric."
    assert bytes_tuple_to_string(string_to_bytes_tuple(target)) == target
