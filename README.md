# 🛡️ Operation Safe Vault — Fraud Shield AI

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.2.2-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Unified Multi-Modal Scam Detection & Real-Time Voice Forensics Platform**
> 
> GitHub Repository: [https://github.com/pragya757/Operationhackthon.git](https://github.com/pragya757/Operationhackthon.git)

---

## 🌟 Key Features

### 🎙️ Audio & Speech Forensics (Voice Clone Detection)
- **Real-Time Live Call Streaming:** WebSocket-based 16 kHz audio streaming with sub-second chunk ingestion.
- **Dual-Engine Threat Fusion:** Combines **Wav2Vec2 Deepfake Detector** (`garystafford/wav2vec2-deepfake-voice-detector`) and custom **Spectrogram CNN** for high-precision synthetic voice classification.
- **Smart Channel Selection:** Automatic fallback between caller tab audio and analyst microphone channels to eliminate silent input delays.
- **Speech-to-Text Transcription:** Real-time ASR powered by **Faster-Whisper** for conversation logging and intent evaluation.

### 🔍 Multi-Modal Scam Detection
- **Text & Intent Classification:** SMS and Email scam detection using NLP intent classifiers, stylometry analysis, and semantic search.
- **URL & Domain Security:** Sandbox execution, WHOIS lookup, SSL header verification, and virus intelligence scanning.
- **File & Malware Analysis:** YARA rules, hash lookup, and payload structure inspection.
- **Email Security Engine:** Header SPF/DKIM/DMARC validation with phishing pattern identification.

### 🛡️ Enterprise Security & Guardrails
- **Shadow Guard Middleware:** Prompt injection defense blocking malicious instruction overrides.
- **DLP Guard:** Data Loss Prevention filter preventing credentials, API keys, and PII exposure.
- **Vector Database Intelligence:** ChromaDB semantic search seeded with 55+ known scam vectors.
- **Human-in-the-Loop Feedback:** Dynamic feedback storage mechanism to continuously fine-tune threat weights.

---

## 🏗️ Architecture & Directory Structure

```text
latest-osv-main/
├── backend/
│   ├── main.py                     # FastAPI server entry point & WebSocket handlers
│   ├── requirements.txt            # Python dependencies (PyTorch, FastAPI, ChromaDB, etc.)
│   ├── core/
│   │   ├── classifier.py          # Central multi-modal classification engine
│   │   ├── threat_score.py        # Threat Fusion scoring & decision policy
│   │   ├── vector_db.py           # ChromaDB vector store integration
│   │   ├── spectrogram_generator.py # Real-time audio spectrogram generator
│   │   └── live_call.py           # Live stream buffer & session manager
│   ├── detectors/
│   │   ├── voice_clone_detector.py # Wav2Vec2 + Spectrogram CNN Threat Fusion
│   │   ├── spectrogram_detector.py # Custom PyTorch CNN spectrogram model
│   │   ├── text_detector.py       # SMS / Email text scam intent detector
│   │   ├── url_detector.py        # Domain heuristics & sandbox URL scanner
│   │   └── email_detector.py      # IMAP & header authentication scanner
│   └── models/
│       └── spectrogram_cnn/       # Trained CNN weights (spectrogram_cnn_whatsapp.pth)
│
├── frontend/
│   ├── package.json               # Next.js 16 dependencies & dev scripts
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx           # Dashboard home
│   │   │   └── voice-clone/      # Voice Forensics & Live Call interface
│   │   └── components/            # UI components (ScoreRing, ResultPanel, Navbar)
└── README.md
```

---

## 💻 Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn, PyTorch (MPS / Metal Accelerated)
- **Frontend:** Next.js 16.2 (Turbopack), React 19, Tailwind CSS, Lucide React
- **ML / AI Models:** Wav2Vec2, Spectrogram CNN, Faster-Whisper, Sentence-Transformers
- **Database:** ChromaDB (Vector Store), SQLite

---

## 🚀 Getting Started

### Prerequisites
- **Python:** `Python 3.13` (Recommended for Apple Silicon thread-safety)
- **Node.js:** `Node.js 18+` & `npm`

---

### 1️⃣ Setting Up & Running the Backend

From the project root directory (`latest-osv-main`):

```bash
# 1. Create Python 3.13 virtual environment
python3.13 -m venv venv313

# 2. Activate virtual environment
source venv313/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Start backend server
cd backend
python3 main.py
```

> **Backend Status:** Server will start at **`http://localhost:8000`**  
> **Interactive API Docs:** **`http://localhost:8000/docs`**

---

### 2️⃣ Setting Up & Running the Frontend

Open a **new terminal tab/window**:

```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Install Node packages
npm install

# 3. Start Next.js development server
npm run dev
```

> **Frontend Dashboard:** App will be live at **`http://localhost:3000`**

---

## 📡 Key API & WebSocket Endpoints

| Endpoint | Protocol | Description |
| :--- | :--- | :--- |
| `GET /` | HTTP | Backend health check & system feature index |
| `POST /api/voice-clone/analyze` | HTTP POST | Audio file upload deepfake analysis |
| `WS /ws/production-live-call/{call_id}` | WebSocket | Real-time audio stream forensics & live call scoring |
| `GET /docs` | HTTP | Interactive OpenAPI Swagger documentation |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
