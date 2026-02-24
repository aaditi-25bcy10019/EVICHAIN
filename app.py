"""
EVICHAIN v2 - Complete Secure Evidence Upload Portal
=====================================================
Team: twinlogic | Cyber Carnival 2026 | VIT Bhopal

Features:
  - RBAC Login (Investigator / Admin)
  - OWASP 10-check file validation
  - Real ML model (Isolation Forest, trained on real files)
  - AES-256-GCM encryption of stored files
  - RSA-2048 digital signature per upload
  - SHA-256 + SHA-3 dual hashing
  - Blockchain-style audit log
  - Claude AI threat explanation
  - Numerical trust score dashboard
"""

import os, json, hashlib, base64, pickle, sqlite3, secrets
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Crypto
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256 as CryptoSHA256
from Crypto.Random import get_random_bytes

# ML
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import requests as http_requests

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["UPLOAD_FOLDER"]        = "uploads"
app.config["MAX_CONTENT_LENGTH"]   = 16 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-6"
MODEL_FILE     = "models/isolation_forest.pkl"
SCALER_FILE    = "models/scaler.pkl"
DB_FILE        = "evichain.db"
RSA_KEY_FILE   = "models/rsa_private.pem"

ALLOWED_EXTENSIONS   = {"pdf", "png", "jpg", "jpeg", "txt", "docx"}
DANGEROUS_EXTENSIONS = {"php","php3","exe","bat","cmd","sh","bash","ps1",
                         "js","asp","aspx","jsp","svg","html","zip","tar","gz"}
MAGIC_BYTES = {
    "pdf":  [b"%PDF"],
    "png":  [b"\x89PNG"],
    "jpg":  [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "docx": [b"PK\x03\x04"],
}
MAX_FILE_SIZE_MB = 10

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'investigator',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            sha3_256 TEXT NOT NULL,
            aes_tag TEXT NOT NULL,
            rsa_signature TEXT NOT NULL,
            file_size_kb REAL NOT NULL,
            threat_score INTEGER NOT NULL,
            ml_verdict TEXT NOT NULL,
            entropy REAL NOT NULL,
            uploaded_by TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id INTEGER,
            action TEXT NOT NULL,
            performed_by TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT,
            prev_hash TEXT,
            log_hash TEXT
        )
    """)

    # Default users
    users = [
        ("admin",        generate_password_hash("admin123"),        "admin"),
        ("investigator", generate_password_hash("invest123"),       "investigator"),
        ("aaditi",       generate_password_hash("twinlogic2026"),   "investigator"),
        ("himisha",      generate_password_hash("twinlogic2026"),   "investigator"),
    ]
    for username, pw_hash, role in users:
        try:
            cur.execute(
                "INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                (username, pw_hash, role, datetime.now(timezone.utc).isoformat())
            )
        except sqlite3.IntegrityError:
            pass

    con.commit()
    con.close()

# ─────────────────────────────────────────────
# RSA KEY SETUP
# ─────────────────────────────────────────────
def load_or_create_rsa_key():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(RSA_KEY_FILE):
        with open(RSA_KEY_FILE, "rb") as f:
            return RSA.import_key(f.read())
    key = RSA.generate(2048)
    with open(RSA_KEY_FILE, "wb") as f:
        f.write(key.export_key())
    print("[OK] RSA-2048 key generated.")
    return key

RSA_KEY = load_or_create_rsa_key()

# ─────────────────────────────────────────────
# ML MODEL LOADING
# ─────────────────────────────────────────────
def load_or_build_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f: model = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
        print("[OK] Loaded trained ML model.")
        return model, scaler, True
    else:
        print("[WARN] No trained model found — using fallback. Run train_model.py first.")
        np.random.seed(42)
        normal = np.column_stack([
            np.random.uniform(1, 5000, 400),
            np.random.uniform(3.5, 7.0, 400),
            np.random.uniform(0, 0.01, 400),
            np.random.uniform(0.7, 1.0, 400),
            np.zeros(400),
            np.zeros(400),
            np.random.uniform(0, 0.3, 400),
            np.random.uniform(0, 0.05, 400),
            np.random.uniform(0.5, 1.0, 400),
        ])
        scaler = StandardScaler()
        X = scaler.fit_transform(normal)
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X)
        return model, scaler, False

ML_MODEL, ML_SCALER, IS_REAL_MODEL = load_or_build_model()

# ─────────────────────────────────────────────
# RBAC HELPERS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Authentication required"}), 401
        if session.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

def get_user(username):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT username, password_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()
    return row

# ─────────────────────────────────────────────
# AUDIT LOG (BLOCKCHAIN-STYLE)
# ─────────────────────────────────────────────
def add_audit_log(evidence_id, action, performed_by, details=""):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    # Get last log hash for chaining
    cur.execute("SELECT log_hash FROM audit_log ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    prev_hash = row[0] if row else "GENESIS"

    timestamp = datetime.now(timezone.utc).isoformat()+ "Z"
    raw = f"{prev_hash}{evidence_id}{action}{performed_by}{timestamp}{details}"
    log_hash = hashlib.sha256(raw.encode()).hexdigest()

    cur.execute("""
        INSERT INTO audit_log (evidence_id, action, performed_by, timestamp, details, prev_hash, log_hash)
        VALUES (?,?,?,?,?,?,?)
    """, (evidence_id, action, performed_by, timestamp, details, prev_hash, log_hash))

    con.commit()
    con.close()
    return log_hash

# ─────────────────────────────────────────────
# CRYPTO — AES + RSA
# ─────────────────────────────────────────────
def aes_encrypt(file_bytes: bytes) -> dict:
    """AES-256-GCM encryption. Returns encrypted bytes + tag + nonce."""
    key   = get_random_bytes(32)   # AES-256
    nonce = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(file_bytes)
    return {
        "ciphertext": ciphertext,
        "key":   base64.b64encode(key).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "tag":   base64.b64encode(tag).decode(),
    }

def rsa_sign(file_bytes: bytes) -> str:
    """RSA-2048 PKCS#1 v1.5 signature of SHA-256 hash of file."""
    h   = CryptoSHA256.new(file_bytes)
    sig = pkcs1_15.new(RSA_KEY).sign(h)
    return base64.b64encode(sig).decode()

def rsa_verify(file_bytes: bytes, signature_b64: str) -> bool:
    """Verify RSA signature."""
    try:
        h   = CryptoSHA256.new(file_bytes)
        sig = base64.b64decode(signature_b64)
        pkcs1_15.new(RSA_KEY).verify(h, sig)
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────
# ML FEATURE EXTRACTION
# ─────────────────────────────────────────────
def compute_entropy(data: bytes) -> float:
    if not data: return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256) / len(data)
    freq = freq[freq > 0]
    return float(-np.sum(freq * np.log2(freq)))

def extract_ml_features(file_bytes: bytes, filename: str, mime_mismatch: bool) -> np.ndarray:
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    risk_map = {"exe":10,"php":9,"sh":8,"bat":8,"js":6,"svg":5,"html":4}
    ext_risk = risk_map.get(ext, 0)
    size_kb  = len(file_bytes) / 1024
    entropy  = compute_entropy(file_bytes)
    null_r   = file_bytes.count(b"\x00") / max(len(file_bytes), 1)
    printable= sum(32 <= b <= 126 for b in file_bytes) / max(len(file_bytes), 1)
    high_b   = sum(b > 127 for b in file_bytes) / max(len(file_bytes), 1)
    ctrl_c   = sum(1 <= b <= 31 for b in file_bytes) / max(len(file_bytes), 1)
    unique_b = len(set(file_bytes)) / 256
    return np.array([[size_kb, entropy, null_r, printable, ext_risk,
                      1 if mime_mismatch else 0, high_b, ctrl_c, unique_b]])

def ml_threat_analysis(file_bytes: bytes, filename: str, mime_mismatch: bool) -> dict:
    features     = extract_ml_features(file_bytes, filename, mime_mismatch)
    scaled       = ML_SCALER.transform(features)
    raw_score    = ML_MODEL.score_samples(scaled)[0]
    threat_score = int(np.clip((-raw_score) * 200, 0, 100))
    confidence   = round(min(abs(raw_score) * 200, 100), 1)

    entropy = float(features[0][1])
    null_r  = float(features[0][2])
    flags   = []

    if entropy > 7.5:
        flags.append("Extremely high entropy — possible encrypted/obfuscated payload")
    elif entropy > 6.5:
        flags.append("Elevated entropy — possible compressed binary content")
    if null_r > 0.05:
        flags.append("High null byte density — possible binary injection")
    if features[0][4] > 5:
        flags.append("High-risk file extension")
    if mime_mismatch:
        flags.append("MIME type mismatch — content spoofing detected")

    return {
        "threat_score":     threat_score,
        "confidence_pct":   confidence,
        "verdict":          "MALICIOUS" if threat_score >= 70 else ("SUSPICIOUS" if threat_score >= 40 else "CLEAN"),
        "entropy":          round(entropy, 4),
        "null_byte_ratio":  round(null_r, 6),
        "printable_ratio":  round(float(features[0][3]), 4),
        "high_byte_ratio":  round(float(features[0][6]), 4),
        "unique_bytes":     int(features[0][8] * 256),
        "raw_if_score":     round(float(raw_score), 6),
        "flags":            flags,
        "model_type":       "trained" if IS_REAL_MODEL else "fallback",
    }

# ─────────────────────────────────────────────
# OWASP VALIDATION
# ─────────────────────────────────────────────
def validate_file(file_bytes: bytes, filename: str) -> dict:
    violations, warnings = [], []

    sanitized = secure_filename(filename)
    if sanitized != filename:
        warnings.append(f"Filename sanitized: '{filename}' → '{sanitized}'")

    if "\x00" in filename:
        violations.append("NULL byte injection in filename — path traversal attempt")

    if "." not in filename:
        violations.append("No file extension — rejected")
        return {"passed": False, "violations": violations, "warnings": warnings, "mime_mismatch": False}

    parts = filename.lower().split(".")
    if len(parts) > 2:
        inner = set(parts[1:-1]) & DANGEROUS_EXTENSIONS
        if inner:
            violations.append(f"Double extension attack: dangerous extension '{list(inner)[0]}' hidden in filename")

    ext = parts[-1]
    if ext in DANGEROUS_EXTENSIONS:
        violations.append(f"Blacklisted extension '.{ext}' — executable/dangerous file type")
    if ext not in ALLOWED_EXTENSIONS:
        violations.append(f"Extension '.{ext}' not in allowed list: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        violations.append(f"File size {size_mb:.2f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")

    mime_mismatch = False
    if ext in MAGIC_BYTES:
        if not any(file_bytes.startswith(sig) for sig in MAGIC_BYTES[ext]):
            violations.append(f"Magic bytes mismatch — file content doesn't match .{ext} signature")
            mime_mismatch = True

    if ext == "pdf":
        if b"/JavaScript" in file_bytes:
            violations.append("PDF contains embedded JavaScript — exploit risk")
        if b"/Launch" in file_bytes:
            violations.append("PDF contains /Launch action — command execution risk")

    return {"passed": len(violations) == 0, "violations": violations,
            "warnings": warnings, "mime_mismatch": mime_mismatch}

# ─────────────────────────────────────────────
# CLAUDE AI
# ─────────────────────────────────────────────
def claude_explain(filename, violations, ml_result):
    prompt = f"""You are a cybersecurity expert at an OWASP forensics lab.
A file upload was analyzed. Provide a structured security report.

File: {filename}
OWASP Violations: {json.dumps(violations)}
ML Analysis: verdict={ml_result['verdict']}, threat_score={ml_result['threat_score']}/100,
             confidence={ml_result['confidence_pct']}%, entropy={ml_result['entropy']},
             flags={ml_result['flags']}

Provide exactly:
1. VERDICT (one sentence, court-ready)
2. ATTACK TYPE (name the specific attack e.g. Double Extension Bypass)
3. OWASP CATEGORY (e.g. A03:2021 Injection)
4. TECHNICAL EXPLANATION (2-3 sentences, what the attacker was trying to do)
5. RECOMMENDATION (what the uploader should do if file is legitimate)

Max 200 words. Be precise and professional."""

    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    payload = {"model": CLAUDE_MODEL, "max_tokens": 500,
               "messages": [{"role": "user", "content": prompt}]}
    try:
        r = http_requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"AI analysis unavailable: {e}"

def claude_content_scan(file_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    mime_map = {"pdf":"application/pdf","png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg"}
    if ext not in mime_map:
        return None
    b64 = base64.standard_b64encode(file_bytes).decode()
    media_type = mime_map[ext]
    if media_type == "application/pdf":
        block = {"type":"document","source":{"type":"base64","media_type":media_type,"data":b64}}
    else:
        block = {"type":"image","source":{"type":"base64","media_type":media_type,"data":b64}}
    prompt = """Forensic content scan. Check for:
1. Malicious scripts, exploits, or hidden payloads
2. Sensitive PII or credential exposure
3. Signs of document forgery or manipulation
4. Overall verdict: SAFE / SUSPICIOUS / DANGEROUS
Be concise, max 150 words."""
    headers = {"Content-Type":"application/json","anthropic-version":"2023-06-01"}
    payload = {"model":CLAUDE_MODEL,"max_tokens":400,
               "messages":[{"role":"user","content":[block,{"type":"text","text":prompt}]}]}
    try:
        r = http_requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"Content scan unavailable: {e}"

# ─────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html",
                           username=session["user"],
                           role=session["role"])

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    data = request.get_json()
    username = data.get("username","").strip()
    password = data.get("password","")
    user = get_user(username)
    if not user or not check_password_hash(user[1], password):
        return jsonify({"error": "Invalid credentials"}), 401
    session["user"] = user[0]
    session["role"] = user[2]
    add_audit_log(None, "LOGIN", username, f"role={user[2]}")
    return jsonify({"success": True, "username": user[0], "role": user[2]})

@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    add_audit_log(None, "LOGOUT", user)
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────
# ROUTES — UPLOAD
# ─────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_bytes = file.read()
    filename   = file.filename
    uploader   = session["user"]

    # 1. OWASP Validation
    validation = validate_file(file_bytes, filename)

    # 2. ML Analysis
    ml = ml_threat_analysis(file_bytes, filename, validation.get("mime_mismatch", False))

    # 3. Dual Hashing
    sha256   = hashlib.sha256(file_bytes).hexdigest()
    sha3_256 = hashlib.sha3_256(file_bytes).hexdigest()

    # 4. RSA Digital Signature
    signature = rsa_sign(file_bytes)

    # 5. Decide outcome
    overall_pass = validation["passed"] and ml["verdict"] != "MALICIOUS"

    # 6. AES Encrypt + Save (only if clean)
    aes_tag = ""
    evidence_id = None
    ai_explanation  = None
    ai_content_scan = None

    if overall_pass:
        encrypted     = aes_encrypt(file_bytes)
        safe_filename = f"{sha256[:16]}_{secure_filename(filename)}"
        save_path     = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(encrypted["ciphertext"])
        aes_tag = encrypted["tag"]

        # Save to DB
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("""
            INSERT INTO evidence
            (filename, original_filename, sha256, sha3_256, aes_tag, rsa_signature,
             file_size_kb, threat_score, ml_verdict, entropy, uploaded_by, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (safe_filename, filename, sha256, sha3_256, aes_tag, signature,
              round(len(file_bytes)/1024, 2), ml["threat_score"], ml["verdict"],
              ml["entropy"], uploader, datetime.now(timezone.utc).isoformat()+"Z"))
        evidence_id = cur.lastrowid
        con.commit()
        con.close()

        add_audit_log(evidence_id, "UPLOAD_ACCEPTED", uploader,
                      f"sha256={sha256[:16]}... threat={ml['threat_score']}")

        # AI content scan for PDFs/images
        ext = filename.rsplit(".",1)[-1].lower() if "." in filename else ""
        if ext in {"pdf","png","jpg","jpeg"}:
            ai_content_scan = claude_content_scan(file_bytes, filename)

    else:
        add_audit_log(None, "UPLOAD_REJECTED", uploader,
                      f"file={filename} violations={len(validation['violations'])}")
        ai_explanation = claude_explain(filename, validation["violations"], ml)

    return jsonify({
        "filename":        filename,
        "file_size_kb":    round(len(file_bytes)/1024, 2),
        "timestamp":       datetime.now(timezone.utc).isoformat()+"Z",
        "overall_result":  "ACCEPTED" if overall_pass else "REJECTED",
        "uploaded_by":     uploader,
        "evidence_id":     evidence_id,
        "owasp_validation": {
            "passed":     validation["passed"],
            "violations": validation["violations"],
            "warnings":   validation["warnings"],
        },
        "ml_analysis":     ml,
        "hashes": {"sha256": sha256, "sha3_256": sha3_256},
        "rsa_signature":   signature[:64] + "...",  # truncated for display
        "aes_encrypted":   overall_pass,
        "ai_explanation":  ai_explanation,
        "ai_content_scan": ai_content_scan,
    })

# ─────────────────────────────────────────────
# ROUTES — ADMIN
# ─────────────────────────────────────────────
@app.route("/api/admin/evidence")
@admin_required
def list_evidence():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM evidence ORDER BY uploaded_at DESC LIMIT 50")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify(rows)

@app.route("/api/admin/audit-log")
@admin_required
def get_audit_log():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 100")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify(rows)

@app.route("/api/admin/users")
@admin_required
def list_users():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT id, username, role, created_at FROM users")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify(rows)

@app.route("/api/stats")
@login_required
def stats():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM evidence")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM evidence WHERE ml_verdict='CLEAN'")
    clean = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM evidence WHERE ml_verdict='MALICIOUS'")
    malicious = cur.fetchone()[0]
    cur.execute("SELECT AVG(threat_score) FROM evidence")
    avg_threat = cur.fetchone()[0] or 0
    cur.execute("SELECT AVG(entropy) FROM evidence")
    avg_entropy = cur.fetchone()[0] or 0
    con.close()
    return jsonify({
        "total_uploads": total,
        "clean": clean,
        "malicious": malicious,
        "suspicious": total - clean - malicious,
        "avg_threat_score": round(avg_threat, 1),
        "avg_entropy": round(avg_entropy, 3),
        "model_type": "trained" if IS_REAL_MODEL else "fallback",
    })

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    init_db()
    port = int(os.environ.get("PORT",5000)
    app.run(host="0.0.0.0, port=port)
