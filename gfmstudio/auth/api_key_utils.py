# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import datetime
import hashlib
import secrets
import string
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from fastapi.exceptions import HTTPException

from gfmstudio.config import settings
from gfmstudio.log import logger

API_KEY_MAX_DURATION = 365


def get_cipher():
    try:
        return Fernet(settings.API_ENCRYPTION_KEY.encode())
    except Exception:
        logger.exception("Define a valid API_ENCRYPTION_KEY in the env variables.")
        raise HTTPException(
            status_code=403,
            detail="Authentication system configuration error. Check server logs.",
        )


def hash_api_key(api_key: str) -> str:
    """Hash api key."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def encrypt_key(plain_key: str) -> str:
    """Encrypts the plain API key using the master key."""
    cipher = get_cipher()
    return cipher.encrypt(plain_key.encode()).decode()


@lru_cache()
def decrypt_key(encrypted_key: str) -> str:
    """Decrypts the stored key back into its plain form."""
    cipher = get_cipher()
    try:
        return cipher.decrypt(encrypted_key.encode()).decode()
    except InvalidToken:
        return encrypted_key


def generate_apikey(key_length: int = 32) -> dict:
    """Generate a random API Key

    Args:
        key_length (int): The length of the random portion of the key.
            Defaults to 32.
        encrypt (bool): Whether to encrypt the key. If True, the function
            requires the `encrypt_key` utility to be functional.

    """
    characters = string.ascii_letters + string.digits
    # pak - partner/public api key
    api_key = f"pak-{''.join(secrets.choice(characters) for _ in range(key_length))}"
    hashed_key = hash_api_key(api_key)
    encrypted_key = encrypt_key(api_key)

    return {
        "api_key": api_key,
        "encrypted_key": encrypted_key,
        "hashed_key": hashed_key,
    }


def apikey_expiry_date(key_duration=90) -> datetime.datetime:
    key_duration = min(key_duration, API_KEY_MAX_DURATION)
    current_date = datetime.datetime.utcnow()
    expiry_date = current_date + datetime.timedelta(days=key_duration)
    return expiry_date
