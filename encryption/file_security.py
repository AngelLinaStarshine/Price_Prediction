import os
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
import base64
import json

def encrypt_data(data, password):
    salt = os.urandom(16)
    key = scrypt(password.encode(), salt, dklen=32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    return json.dumps({
        'salt': base64.b64encode(salt).decode(),
        'nonce': base64.b64encode(cipher.nonce).decode(),
        'tag': base64.b64encode(tag).decode(),
        'ciphertext': base64.b64encode(ciphertext).decode()
    })

def decrypt_data(encrypted_data, password):
    data = json.loads(encrypted_data)
    salt = base64.b64decode(data['salt'])
    nonce = base64.b64decode(data['nonce'])
    tag = base64.b64decode(data['tag'])
    ciphertext = base64.b64decode(data['ciphertext'])

    key = scrypt(password.encode(), salt, dklen=32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)
