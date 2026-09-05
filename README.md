# 📖 ScriptureLM-SelfBuilt

> **A self-built Bible Language Model + Retrieval-Augmented Generation (RAG) system.**
> The Transformer language model was implemented and trained from scratch, while a pretrained embedding model is used for semantic retrieval.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-V4_Causal_Transformer-6E56CF?style=flat-square" alt="Model V4"/>
  <img src="https://img.shields.io/badge/Params-~5.85M-orange?style=flat-square" alt="Params"/>
  <img src="https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-4E9F3D?style=flat-square" alt="Embeddings"/>
  <img src="https://img.shields.io/badge/Verses-31%2C102-b3541e?style=flat-square" alt="Verses"/>
  <img src="https://img.shields.io/badge/License-Unspecified-lightgrey?style=flat-square" alt="License"/>
</p>

<p align="center">
  <a href="https://scripturelm-selfbuilt.onrender.com"><b>🚀 Live Demo</b></a>
  &nbsp;•&nbsp;
  <a href="https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt"><b>💻 GitHub Repository</b></a>
</p>

---

## 📌 About the Project

**ScriptureLM-SelfBuilt** is an educational, domain-specific NLP project that explores building an entire language-model pipeline **from the ground up** — no shortcuts, no pretrained generative APIs for the final answer.

The project is really two related pipelines, applied to the Bible as a text corpus:

**1. Model development pipeline**
```
Bible Corpus → BPE Tokenizer → Dataset → Transformer → Training → QA Fine-Tuning
```

**2. Production application pipeline**
```
User Question → Embedding → RAG Retrieval → Grounded Answer Builder → Flask → Web UI
```

> **Note:** No external generative AI API (e.g. OpenAI, Gemini, Claude) is used to produce the final answer. The production answer is built entirely from **retrieval + grounding logic**, using a pretrained embedding model for search over passages drawn from the self-built pipeline above.

---

## ✨ Key Highlights

| Category | Detail |
|---|---|
| 📚 Corpus | 31,102 Bible verses |
| 🔤 Tokenizer | Custom BPE — Vocab: **2,048**, Merges: **1,985** |
| 🧠 Model | Self-built causal Transformer — **V4** |
| ⚙️ Parameters | ~**5.85M** |
| 🧱 Layers | 6 Transformer layers |
| 👀 Attention Heads | 8 |
| 📐 Embedding Dimension | 256 |
| 📏 Context Length | 256 |
| 🔎 RAG Embeddings | 384-dimensional (`all-MiniLM-L6-v2`) |
| ❓ QA Examples | 12,000 |
| 🌐 Serving | Flask + REST API |
| ❤️ Health Check | `/health` health-check endpoint for production deployment |
| ☁️ Hosting | Render — free tier |

---

## 🏗 System Architecture

**Production request path (what actually runs when a user asks a question):**

```
┌─────────────────────┐
│     User Question    │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  all-MiniLM-L6-v2     │   (pretrained embedding model)
│  Question Embedding   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Semantic Retrieval   │
│  (31,102 × 384 index) │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Book Diversity +     │
│  Top 5 Passages       │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Grounded Answer Builder│
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│      Flask API        │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│    Web Interface       │
└────────────────────────┘
```

**Offline model-development path (built and trained separately, not called at request time):**

```
Bible Corpus → BPE Tokenizer → Dataset → Self-Built Transformer (V4) → Training → QA Fine-Tuning
```

> 💡 The self-built Transformer is a **model-development artifact** — it was trained and evaluated separately. It is not invoked in the live production request path above; production answers are grounded using retrieval with the pretrained `all-MiniLM-L6-v2` embedding model.

---

## 📚 Dataset

- **Location:** `data/raw/bible_corpus.txt`
- **Size:** 31,102 verses
- **Format:** One verse per line, in `[Book Chapter:Verse] verse text` format

**Example corpus format:**

```
[Genesis 1:1] In the beginning God created the heaven and the earth.
[Genesis 1:2] And the earth was without form, and void...
[John 3:16] For God so loved the world, that he gave his only begotten Son...
```

---

## 🔤 Tokenizer

A **custom Byte Pair Encoding (BPE)** tokenizer converts raw scripture text into token IDs the Transformer can process.

| Property | Value |
|---|---|
| Vocabulary Size | 2,048 |
| BPE Merges | 1,985 |
| File | `data/processed/tokenizer_v4.json` |

**Tokenizer Workflow:**

```
Raw Text
   │
   ▼
Pre-tokenization (split into base symbols)
   │
   ▼
Iterative BPE Merges (1,985 merges)
   │
   ▼
Vocabulary (2,048 tokens)
   │
   ▼
Token IDs → fed into Transformer
```

---

## 🧠 Transformer Model

The **V4 causal Transformer** is trained from scratch on the tokenized Bible corpus.

| Spec | Value |
|---|---|
| Vocabulary Size | 2,048 |
| Context Length | 256 |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| Dropout | 0.1 |
| Parameters | ~5.85M |

**Core components:**

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward networks
- Residual connections
- Pre-normalization
- LayerNorm
- Causal language-model head

**Transformer Flow:**

```
Input Tokens
     │
     ▼
Token Embeddings + Positional Embeddings
     │
     ▼
┌────────────────────────────┐
│   × 6 Transformer Blocks    │
│  ┌───────────────────────┐  │
│  │ LayerNorm (Pre-Norm)   │  │
│  │ Multi-Head Attention   │  │
│  │ Residual Connection    │  │
│  │ LayerNorm (Pre-Norm)   │  │
│  │ Feed-Forward Network   │  │
│  │ Residual Connection    │  │
│  └───────────────────────┘  │
└────────────────────────────┘
     │
     ▼
Final LayerNorm
     │
     ▼
Causal LM Head → Next-token logits
```

---

## 🏋️ Training

| Parameter | Value |
|---|---|
| Batch Size | 2 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 32 |
| Training Steps | 5,000 |
| Initial Learning Rate | 3e-4 |
| Final Learning Rate | 3e-5 |
| Warmup Steps | 300 |
| Optimizer | AdamW |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Dropout | 0.1 |
| Scheduler | Warmup + Cosine Decay |

> ✅ **Best V4 validation loss:** `3.85433`

**Training script:** `src/train_v4.py`

---

## 🔎 RAG System

Retrieval-Augmented Generation grounds every answer in **actual retrieved Bible passages** rather than free-form model generation.

**Pipeline:**

1. Document creation — `src/rag/build_documents.py` → `data/rag/documents.json`
2. Embedding generation — `all-MiniLM-L6-v2` → `data/rag/embeddings.npy` (**31,102 × 384**)
3. Semantic retrieval via `src/rag/retriever.py`
4. Candidate retrieval
5. Book diversity selection
6. Top 5 passages selected
7. Grounded answer builder assembles the final response

**RAG Workflow:**

```
User Question
     │
     ▼
Question Embedding (all-MiniLM-L6-v2)
     │
     ▼
Semantic Similarity Search (31,102 × 384 embeddings)
     │
     ▼
Candidate Retrieval
     │
     ▼
Book Diversity Selection
     │
     ▼
Top 5 Passages
     │
     ▼
Grounded Answer Builder
```

> ⚠️ **Important distinction:** `all-MiniLM-L6-v2` is a **pretrained embedding model** used only for semantic search — it is separate from the self-built causal Transformer described above.

---

## ❓ QA Fine-Tuning

An experimental QA fine-tuning pipeline was built on top of the V4 Transformer.

| Split | Count |
|---|---|
| Training | 9,648 |
| Validation | 1,166 |
| Testing | 1,186 |
| **Total** | **12,000** |

**Dataset:** `data/rag/qa_dataset_v4.json`

- Trained using **answer-only loss masking**
- Marked **experimental** because:
  - The Transformer is relatively small (~5.85M parameters)
  - It is **not instruction-tuned**
  - Direct free-form QA generation is limited in reliability

> ✅ **Best validation answer loss:** `2.667409`

Because of these limitations, **production answers rely on retrieval + grounded answer building**, not direct free-form generation from the QA-fine-tuned model.

---

## ⚙️ How the Application Works

```
 1. User enters a Bible question
 2. Question becomes an embedding
 3. Semantic similarity search runs
 4. Relevant Bible verses are retrieved
 5. Results are diversified across books
 6. Top 5 passages are selected
 7. Relevant sentences are selected
 8. Grounded answer is built
 9. Flask returns the response
10. Sources and similarity scores are displayed
```

```
[ User Input ] → [ Embedding ] → [ Similarity Search ] → [ Retrieval ]
       → [ Book Diversification ] → [ Top-5 Passages ] → [ Sentence Selection ]
       → [ Grounded Answer Builder ] → [ Flask Response ] → [ UI Display ]
```

---

## 🌐 Web Application

- **Backend:** Flask (`app.py`)
- **Frontend:** HTML/CSS/JavaScript (`ui/templates/index.html`, `ui/static/`)
- The frontend sends the user's question to the Flask backend via `/api/ask` and renders the returned answer, sources, and scores.

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/api/ask` | Accepts a question, returns a grounded answer with sources |
| `GET` | `/health` | Health-check endpoint for production deployment |

**Example request — `POST /api/ask`:**

```json
{
  "question": "What does the Bible say about love?"
}
```

**Example response structure** *(illustrative — actual values depend on the live retrieval result)*:

```json
{
  "question": "What does the Bible say about love?",
  "answer": "...",
  "sources": [
    {
      "reference": "...",
      "text": "...",
      "score": 0.0
    }
  ]
}
```

**Health check response:**

```json
{
  "status": "ok"
}
```

---

## 🗂 Project Structure

```
ScriptureLM-SelfBuilt/
├── app.py
├── README.md
├── requirements.txt
├── .python-version
├── .gitignore
├── data/
│   ├── raw/
│   │   └── bible_corpus.txt
│   ├── processed/
│   │   ├── dataset_v4.pt
│   │   └── tokenizer_v4.json
│   └── rag/
│       ├── documents.json
│       └── embeddings.npy
├── src/
│   ├── dataset_v4.py
│   ├── generate_v4.py
│   ├── model_v4.py
│   ├── tokenizer_v4.py
│   ├── train_v4.py
│   └── rag/
│       ├── build_documents.py
│       ├── build_embeddings.py
│       ├── build_qa_dataset_v4.py
│       ├── retriever.py
│       ├── test_rag_qa_v4.py
│       └── train_qa_v4.py
└── ui/
    ├── static/
    └── templates/
        └── index.html
```

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| PyTorch | Transformer implementation and training |
| NumPy | Numerical processing |
| Sentence Transformers | Embeddings |
| all-MiniLM-L6-v2 | Semantic retrieval |
| Flask | Web application and REST API |
| Gunicorn | Production WSGI server |
| HTML/CSS/JavaScript | Frontend |
| Git | Version control |
| GitHub | Repository |
| Render | Deployment |

---

## 💻 Run Locally

```bash
git clone https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt.git
cd ScriptureLM-SelfBuilt
```

**Create a virtual environment (Windows):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the app:**

```bash
python app.py
```

| Resource | URL |
|---|---|
| Local App | http://127.0.0.1:5000 |
| Local Health Check | http://127.0.0.1:5000/health |

---

## ☁️ Deployment

Deployed on **Render**, using the free hosting tier.

| Setting | Value |
|---|---|
| Runtime | Python 3.12 |
| PyTorch | CPU-only |
| Server | Gunicorn |
| Dependencies | Pinned in `requirements.txt` |

**Build command:**

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
gunicorn app:app
```

| Resource | URL |
|---|---|
| Live App | https://scripturelm-selfbuilt.onrender.com |
| Health Check | https://scripturelm-selfbuilt.onrender.com/health |

---

## 💬 Example Questions

- What does the Bible say about love?
- Who was Moses?
- What does the Bible say about faith?
- What does the Bible say about forgiveness?
- What does the Bible say about hope?

---

## ✅ Testing / Validation

- RAG testing script: `src/rag/test_rag_qa_v4.py`
- Retrieval tested on topics such as *love* and *Moses*
- Best V4 validation loss: `3.85433`
- Best QA validation answer loss: `2.667409`

---

## 🎯 Why I Built This

The goal was to genuinely **understand the internals and workflow of LLM systems** — not simply call an existing chatbot API and wrap a UI around it.

This project covers, hands-on:

- Dataset preparation
- Tokenization
- Byte Pair Encoding (BPE)
- Embeddings
- Self-attention
- Transformers
- Training
- Fine-tuning
- Semantic search
- Retrieval-Augmented Generation (RAG)
- REST APIs
- Flask
- Deployment

---

## 📋 Project Status

| Component | Status |
|---|:---:|
| Bible corpus | ✅ |
| BPE tokenizer | ✅ |
| Transformer V4 | ✅ |
| V4 training | ✅ |
| QA dataset | ✅ |
| QA fine-tuning | ✅ |
| Bible documents | ✅ |
| Semantic embeddings | ✅ |
| RAG retrieval | ✅ |
| Grounded answer builder | ✅ |
| Flask application | ✅ |
| REST API | ✅ |
| Health check | ✅ |
| Render deployment | ✅ |
| GitHub repository | ✅ |
| Documentation | ✅ |

---

## 👤 Author

**Samuel D**

B.E. Computer Science and Engineering

Prathyusha Engineering College

Graduating 2027

GitHub: [github.com/JOSESAMUEL14](https://github.com/JOSESAMUEL14)

---

<div align="center">

### Scripture · Intelligence · Retrieval · From Scratch

*"Built from Scripture.*
*Built to understand LLMs.*
*Built with a self-trained Transformer."*
</div>

---

## 📄 License

No license has been specified for this repository.
