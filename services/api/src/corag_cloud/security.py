"""Password hashing for the self-serve identity layer.

Stored format: ``scrypt$<salt_b64>$<hash_b64>`` with the OWASP baseline work
factor (N=2**14, r=8, p=1). ``verify_password`` is a constant-time mirror.
"""

import base64
import hashlib
import os
import secrets

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DKLEN = 32


def hash_password(password: str) -> str:
    """Return ``scrypt$<salt_b64>$<hash_b64>`` for a plaintext password."""
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return (
        f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"
    )


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time verification of a plaintext against a stored hash."""
    if not stored:
        return False
    parts = stored.split("$", 2)
    if len(parts) != 3 or parts[0] != "scrypt":
        return False
    try:
        # binascii.Error (malformed base64) subclasses ValueError.
        salt = base64.b64decode(parts[1])
        expected = base64.b64decode(parts[2])
    except ValueError:
        return False
    derived = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=len(expected),
    )
    return secrets.compare_digest(derived, expected)
