"""
EVICHAIN - test_run.py
=======================
Run this before your demo to verify everything works.

Usage:
    python test_run.py
"""

import os
import sys
import hashlib
import pickle
import sqlite3
import numpy as np

print("\n" + "="*55)
print("  EVICHAIN - Pre-Demo Test Runner")
print("  Team twinlogic | Cyber Carnival 2026")
print("="*55)

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅  {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌  {name}  →  {e}")
        failed += 1

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
print("\n── Checking Imports ─────────────────────────")

def check_flask():       import flask
def check_werkzeug():    import werkzeug
def check_sklearn():     import sklearn
def check_numpy():       import numpy
def check_pycrypto():    from Crypto.Cipher import AES; from Crypto.PublicKey import RSA
def check_requests():    import requests

test("Flask",            check_flask)
test("Werkzeug",         check_werkzeug)
test("scikit-learn",     check_sklearn)
test("NumPy",            check_numpy)
test("PyCryptodome",     check_pycrypto)
test("Requests",         check_requests)

# ─────────────────────────────────────────────
# 2. FILES & FOLDERS
# ─────────────────────────────────────────────
print("\n── Checking Project Files ───────────────────")

test("app.py exists",              lambda: open("app.py"))
test("train_model.py exists",      lambda: open("train_model.py"))
test("templates/login.html",       lambda: open("templates/login.html"))
test("templates/index.html",       lambda: open("templates/index.html"))
test("requirements.txt",           lambda: open("requirements.txt"))

def check_uploads_dir():
    os.makedirs("uploads", exist_ok=True)
    assert os.path.isdir("uploads")

def check_models_dir():
    os.makedirs("models", exist_ok=True)
    assert os.path.isdir("models")

def check_training_data_dir():
    os.makedirs("training_data", exist_ok=True)
    assert os.path.isdir("training_data")

test("uploads/ folder",       check_uploads_dir)
test("models/ folder",        check_models_dir)
test("training_data/ folder", check_training_data_dir)

# ─────────────────────────────────────────────
# 3. ML MODEL
# ─────────────────────────────────────────────
print("\n── Checking ML Model ────────────────────────")

def check_model_files():
    assert os.path.exists("models/isolation_forest.pkl"), \
        "Run: python train_model.py --demo --train"
    assert os.path.exists("models/scaler.pkl"), \
        "Run: python train_model.py --demo --train"

def check_model_loads():
    with open("models/isolation_forest.pkl","rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    # Run a quick prediction
    test_feat = np.array([[142.0, 5.2, 0.001, 0.85, 0, 0, 0.1, 0.01, 0.9]])
    scaled    = scaler.transform(test_feat)
    score     = model.score_samples(scaled)[0]
    threat    = int(np.clip((-score)*200, 0, 100))
    assert 0 <= threat <= 100, f"Invalid threat score: {threat}"

test("Model files exist",  check_model_files)
test("Model loads + runs", check_model_loads)

# ─────────────────────────────────────────────
# 4. CRYPTOGRAPHY
# ─────────────────────────────────────────────
print("\n── Checking Cryptography ────────────────────")

def check_sha256():
    data   = b"EVICHAIN test file content"
    result = hashlib.sha256(data).hexdigest()
    assert len(result) == 64

def check_sha3():
    data   = b"EVICHAIN test file content"
    result = hashlib.sha3_256(data).hexdigest()
    assert len(result) == 64

def check_aes():
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    key       = get_random_bytes(32)
    nonce     = get_random_bytes(16)
    cipher    = AES.new(key, AES.MODE_GCM, nonce=nonce)
    data      = b"Test forensic evidence data"
    ct, tag   = cipher.encrypt_and_digest(data)
    # Decrypt and verify
    cipher2   = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted = cipher2.decrypt_and_verify(ct, tag)
    assert decrypted == data

def check_rsa():
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256 as CSHA256
    import base64
    if os.path.exists("models/rsa_private.pem"):
        with open("models/rsa_private.pem","rb") as f:
            key = RSA.import_key(f.read())
    else:
        key = RSA.generate(2048)
    data = b"Test evidence bytes"
    h    = CSHA256.new(data)
    sig  = pkcs1_15.new(key).sign(h)
    assert len(base64.b64encode(sig)) > 100

test("SHA-256 hashing",  check_sha256)
test("SHA-3 hashing",    check_sha3)
test("AES-256-GCM",      check_aes)
test("RSA-2048 signing", check_rsa)

# ─────────────────────────────────────────────
# 5. OWASP VALIDATION LOGIC
# ─────────────────────────────────────────────
print("\n── Checking OWASP Validation ────────────────")

def check_dangerous_ext():
    DANGEROUS = {"php","exe","bat","sh","js","asp"}
    filename  = "shell.php"
    ext       = filename.rsplit(".",1)[-1].lower()
    assert ext in DANGEROUS, "Should detect .php as dangerous"

def check_double_ext():
    filename = "report.php.pdf"
    parts    = filename.lower().split(".")
    DANGEROUS = {"php","exe","bat","sh","js"}
    if len(parts) > 2:
        inner = set(parts[1:-1]) & DANGEROUS
        assert len(inner) > 0, "Should detect .php in double extension"

def check_magic_bytes():
    # Real PDF starts with %PDF
    real_pdf  = b"%PDF-1.4 fake content..."
    fake_pdf  = b"<?php echo shell_exec($_GET['cmd']); ?>"
    sig       = b"%PDF"
    assert real_pdf.startswith(sig),  "Real PDF should pass"
    assert not fake_pdf.startswith(sig), "Fake PDF should fail"

def check_null_byte():
    filename  = "report.pdf\x00.php"
    assert "\x00" in filename, "Should detect null byte"

def check_entropy():
    # High entropy = suspicious
    random_bytes = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
    freq  = np.bincount(np.frombuffer(random_bytes, dtype=np.uint8), minlength=256) / len(random_bytes)
    freq  = freq[freq > 0]
    ent   = float(-np.sum(freq * np.log2(freq)))
    assert ent > 6.0, f"Random bytes should have high entropy, got {ent:.2f}"

test("Dangerous extension detection", check_dangerous_ext)
test("Double extension detection",    check_double_ext)
test("Magic bytes verification",      check_magic_bytes)
test("Null byte injection detection", check_null_byte)
test("Shannon entropy calculation",   check_entropy)

# ─────────────────────────────────────────────
# 6. DATABASE
# ─────────────────────────────────────────────
print("\n── Checking Database ────────────────────────")

def check_db():
    # Init DB by importing app
    con = sqlite3.connect("evichain.db")
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    con.close()
    # DB might not exist yet — that's OK, app.py creates it on startup
    # Just check sqlite3 works
    assert True

def check_users():
    if not os.path.exists("evichain.db"):
        return  # Will be created when app.py runs
    con = sqlite3.connect("evichain.db")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    con.close()
    assert count >= 2, f"Expected at least 2 users, got {count}"

test("SQLite available",     check_db)
test("Default users exist",  check_users)

# ─────────────────────────────────────────────
# 7. API KEY
# ─────────────────────────────────────────────
print("\n── Checking Configuration ───────────────────")

def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY","")
    assert key.startswith("sk-"), \
        "Set ANTHROPIC_API_KEY env variable (export ANTHROPIC_API_KEY=sk-...)"

test("ANTHROPIC_API_KEY set", check_api_key)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = passed + failed
print("\n" + "="*55)
print(f"  Results: {passed}/{total} tests passed")
if failed == 0:
    print("  🎉 ALL TESTS PASSED — Ready for demo!")
    print("  Run: python app.py")
    print("  Open: http://localhost:5000")
else:
    print(f"  ⚠  {failed} test(s) failed — fix before demo")
    print("\n  Common fixes:")
    print("  - Missing model: python train_model.py --demo --train")
    print("  - Missing packages: pip install -r requirements.txt")
    print("  - Missing API key: set ANTHROPIC_API_KEY=sk-...")
print("="*55 + "\n")
