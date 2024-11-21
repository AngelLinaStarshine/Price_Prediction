import os
import hashlib
import base64


def generate_random_sha256_key():

    random_bytes = os.urandom(32)  

    sha256_hash = hashlib.sha256(random_bytes).digest()

    encoded_key = base64.b64encode(sha256_hash).decode('utf-8')

    return encoded_key
