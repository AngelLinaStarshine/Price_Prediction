from flask import Flask, jsonify
import secrets
import hashlib

app = Flask(__name__)

def generate_sha256_key():
    random_bytes = secrets.token_bytes(32)
    return hashlib.sha256(random_bytes).hexdigest()

@app.route('/generate-key', methods=['GET'])
def generate_key():
    key = generate_sha256_key()
    return jsonify({'key': key})

if __name__ == "__main__":
    app.run(debug=True)
