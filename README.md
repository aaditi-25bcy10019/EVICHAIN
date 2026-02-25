EVICHAIN
EVICHAIN:Advance secure evidence upload and integrity management system

Overview
EVICHAIN is a secure digital evidence upload and analysis system designed to prevent malicious file submissions, detect statistical anomalies using machine learning, ensure cryptographic integrity, and maintain a tamper-proof audit trail.
The system is built to simulate a real-world forensic intake pipeline where every uploaded file is validated, analyzed, secured, and permanently logged.

Problem Statement
Digital evidence submission systems are vulnerable to:
Malicious file uploads disguised as legitimate documents
Double extension bypass attacks
Embedded scripts inside PDFs
MIME type spoofing
Evidence tampering after upload
Poor logging and lack of audit integrity

EVICHAIN addresses these issues using a layered security architecture.
Architecture Overview

Architecture Overview
Every uploaded file passes through five sequential layers:
OWASP-Based Validation:
Machine Learning Threat Analysis
Cryptographic Protection
AI-Based Explanation Layer
Tamper-Evident Audit Logging

If any critical validation fails, the file is rejected and logged.

Layer 1 – OWASP Validation
Rule-based validation performs multiple security checks:
Filename sanitization
Null byte injection detection
Extension whitelist enforcement
Dangerous extension blacklist
Double extension detection
Magic byte verification
MIME type verification
File size restriction
Embedded JavaScript detection in PDFs
/Launch command detection
The system never trusts file extensions alone. Magic byte and MIME validation are mandatory.

Layer 2 – Machine Learning Threat Detection
EVICHAIN uses an Isolation Forest model for anomaly detection.
Instead of signature-based detection, it analyzes statistical properties of files.
Extracted Features:
File size
Shannon entropy
Null byte ratio
Printable character ratio
High byte ratio
Control character ratio
Unique byte ratio
Extension risk score
MIME mismatch flag
Files are scored from 0–100:
0–39 → Clean
40–69 → Suspicious
70–100 → Malicious
This provides measurable forensic justification.

Layer 3 – Cryptographic Protection
All accepted files undergo:
SHA-256 hashing
SHA3-256 hashing
RSA-2048 digital signature
AES-256-GCM encryption
Security guarantees:
Integrity verification
Authenticity proof
Confidentiality of stored evidence
Tamper detection
No plaintext evidence is stored on disk.

Layer 4 – AI Explanation Engine
If a file is rejected, the system generates:
Attack type explanation
Relevant OWASP category
Technical reasoning
Suggested corrective action
If accepted, content is scanned for:
Hidden scripts
Exploit patterns
Sensitive data exposure
Document forgery indicators
This bridges the gap between technical analysis and human-readable reporting.

Layer 5 – Blockchain-Style Audit Log
Each action is recorded in a chained log format.
Every entry contains:
Previous hash
Action
User
Timestamp
If any log entry is modified, the chain integrity breaks.
This ensures forensic-grade traceability.

Role-Based Access Control

Two roles are implemented:
Investigator:
Upload files
View own results

Admin:
View all evidence
Access audit logs
View all users
Passwords are hashed using bcrypt.

Technology Stack
Backend:
Python
Flask
Security:
hashlib
PyCryptodome
python-magic
Machine Learning:
scikit-learn
NumPy
Database:
SQLite
Deployment:
Render

Key Security Principles Implemented
Defense in depth
Zero trust file validation
Statistical anomaly detection
Cryptographic integrity enforcement
Principle of least privilege
Tamper-evident logging

Future Improvements
Integration with VirusTotal API
PostgreSQL migration
Advanced malware sandboxing
Real blockchain anchoring
Multi-factor authentication
Disclaimer
This project is developed for academic and research purposes in the field of Cyber Security and Digital Forensics.
