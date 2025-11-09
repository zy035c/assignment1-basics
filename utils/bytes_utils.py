from __future__ import annotations

def string_to_bytes_list(string: str) -> list[bytes]:
    return [bytes([bb]) for bb in string.encode('utf-8')]

def string_to_bytes_tuple(string: str) -> tuple[bytes, ...]:
    return tuple(bytes([bb]) for bb in string.encode('utf-8'))

def bytes_tuple_to_string(bytes_tuple: tuple[bytes, ...]) -> str:
    return b"".join(bytes_tuple).decode()

def bytes_list_to_string(bytes_list: list[bytes]) -> str:
    return b"".join(bytes_list).decode()

if __name__ == "__main__":
    target = "Yes, indeed. It is called Lothric."
    assert bytes_tuple_to_string(string_to_bytes_tuple(target)) == target
