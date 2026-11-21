"""Developer API key generation and hashing.

Keys look like ``corag_live_<48 hex chars>``. Only a salted SHA-256 hash is
stored; the plaintext is shown exactly once at creation. Lookup is by exact
hash match, so no prefix scan is needed — the stored prefix is display-only.
"""

import hashlib
import secrets
from dataclasses import dataclass

KEY_PREFIX = "corag_live_"
KEY_SECRET_BYTES = 24
PREFIX_DISPLAY_CHARS = 16
MAX_KEYS_PER_TENANT = 5


@dataclass(frozen=True)
class GeneratedKey:
    plaintext: str
    key_hash: str
    key_prefix: str


def hash_api_key(key: str, salt: str) -> str:
    """Salted SHA-256 of the full key string."""
    return hashlib.sha256(f"{key}{salt}".encode()).hexdigest()


def generate_api_key(salt: str) -> GeneratedKey:
    """Create a new key; the plaintext leaves this process exactly once."""
    plaintext = f"{KEY_PREFIX}{secrets.token_hex(KEY_SECRET_BYTES)}"
    return GeneratedKey(
        plaintext=plaintext,
        key_hash=hash_api_key(plaintext, salt),
        key_prefix=plaintext[:PREFIX_DISPLAY_CHARS],
    )
