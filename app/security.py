"""Password hashing with PBKDF2-SHA256 from Python's standard library.

Stored form:  pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>
A random salt per user means two equal passwords give different hashes; many
iterations make each guess slow for an attacker who steals the table.
"""
import hashlib
import hmac
import secrets

ITERATIONS = 600_000      # OWASP recommendation for PBKDF2-SHA256


def hash_password(password, iterations=ITERATIONS):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        algorithm, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    # compare_digest takes the same time wherever the first difference is
    return hmac.compare_digest(check.hex(), digest)
