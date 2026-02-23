# EVICHAIN — Secure File Upload Portal
**Team:** twinlogic | **Event:** Cyber Carnival 2026 | **VIT Bhopal**

## What It Does

A forensic-grade secure file upload portal with three layers of protection:

1. **OWASP Validation** — 10 security checks on every upload
2. **ML Threat Detection** — Isolation Forest model scores file anomalies 0–100
3. **Claude AI** — Explains rejections in plain English + scans file content

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# On Linux, also install libmagic:
sudo apt-get install libmagic1

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 3. Run the app
python app.py
```

Open http://localhost:5000 in your browser.

---

## OWASP Security Checks

| Check | Attack Prevented |
|---|---|
| Extension whitelist | Unrestricted file upload |
| Dangerous extension blacklist | PHP/EXE/shell upload |
| Double extension detection | shell.php.jpg bypass |
| Magic bytes verification | MIME spoofing |
| MIME type validation | Content-type spoofing |
| Null byte in filename | Path traversal |
| File size limit | DoS / disk exhaustion |
| Embedded JS in PDF | PDF exploit |
| /Launch action in PDF | Command execution |
| Filename sanitization | Directory traversal |

---

## ML Model

Uses **Isolation Forest** trained on normal file feature vectors:
- File size
- Shannon entropy (high = suspicious)
- Null byte ratio
- Printable character ratio
- Extension risk score
- MIME mismatch flag

Score 0–39: CLEAN | 40–69: SUSPICIOUS | 70–100: MALICIOUS

---

## AI Integration (Claude API)

- **Rejection Explanation**: When a file fails, Claude explains the attack type, OWASP category, and what the user should do
- **Content Scanning**: For accepted PDFs/images, Claude scans for hidden threats, PII, and suspicious patterns

---

## Project Structure

```
evichain/
├── app.py              # Flask backend (validation + ML + AI)
├── templates/
│   └── index.html      # Dark cybersecurity UI
├── requirements.txt
├── uploads/            # Saved clean files
└── README.md
```