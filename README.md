📖 ScriptureLM-SelfBuilt
https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-https://scripturelm--selfbuilt.onrender.com-2ea44f?style=flat-square&logo=render&logoColor=white
https://img.shields.io/badge/%F0%9F%93%82_Repository-View_on_GitHub-181717?style=flat-square&logo=github&logoColor=white
https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white
https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white
https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white
https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white

A self-built Bible Language Model + Retrieval-Augmented Generation (RAG) System built entirely from scratch — no external generative AI APIs used.

🌐 Live Demo
Platform	URL	Status
Render (24/7)	https://scripturelm-selfbuilt.onrender.com	✅ Always Live
Health Check	https://scripturelm-selfbuilt.onrender.com/health	✅ Operational
🛠️ Tech Stack
Tool	Purpose	Version
Python	Core Development	3.12
PyTorch	Transformer Implementation & Training	Latest
Sentence Transformers	Embedding Generation	all-MiniLM-L6-v2
Flask	Web Application & REST API	Latest
Gunicorn	Production WSGI Server	Latest
NumPy	Numerical Processing	Latest
HTML/CSS/JS	Frontend Interface	-
Render	Cloud Deployment	Free Plan
✨ Key Highlights
📖 31,102 Bible verses — Complete corpus for training

✏️ Custom BPE Tokenizer — 2,048 vocabulary, 1,985 merges

🧠 Self-built Transformer V4 — ~5.85M parameters, 6 layers, 8 heads

🎯 5,000 Training Steps — Validation loss: 3.85433

📊 12,000 QA Examples — Fine-tuned for Bible questions

🔍 RAG System — all-MiniLM-L6-v2 embeddings (31,102 × 384)

🌐 Flask Web App — Interactive Bible question-answering

📡 REST API — /api/ask, /health endpoints

🚀 Live Deployment — Always accessible on Render

📑 Table of Contents
How It Works

System Architecture

Dataset & Tokenizer

Transformer Model

Training Configuration

RAG System

QA Fine-Tuning

API Reference

Quick Start

Project Structure

Deployment

Example Questions

Limitations

Why I Built This

Author

🔄 How It Works







Complete Pipeline:

text
Bible Corpus → BPE Tokenizer → Tokenized Dataset → Transformer → Training → 
QA Fine-Tuning → Embeddings → RAG → Grounded Answer Builder → Flask → Web Interface
⚠️ Important: This system uses NO external generative AI API for the final answer. The Transformer is self-built, and all responses are grounded in retrieved Bible passages.

🧠 System Architecture
Two-Path Design: Generation + Retrieval
text
┌──────────────────────────────────────────────────────────────────────┐
│                    SCRIPTURELM-SELFBUILT ARCHITECTURE               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │   LANGUAGE MODEL PATH    │    │         RAG PATH             │   │
│  │                          │    │                              │   │
│  │  Bible Corpus           │    │  Document Builder            │   │
│  │  (31,102 verses)        │    │  └───► documents.json        │   │
│  │         ↓               │    │                              │   │
│  │  BPE Tokenizer          │    │  all-MiniLM-L6-v2            │   │
│  │  (Vocab: 2,048)         │    │  └───► embeddings.npy        │   │
│  │         ↓               │    │         (31,102 × 384)       │   │
│  │  Tokenized Dataset      │    │                              │   │
│  │         ↓               │    │  Retriever                   │   │
│  │  Transformer V4         │    │  └───► Top 20 Candidates     │   │
│  │  (6 Layers, 8 Heads)    │    │                              │   │
│  │         ↓               │    │  Book Diversity Selection    │   │
│  │  Training (5,000 steps) │    │  └───► Top 5 Passages        │   │
│  │  └───► Loss: 3.85433    │    │                              │   │
│  │         ↓               │    │                              │   │
│  │  QA Fine-Tuning         │    │                              │   │
│  │  (12,000 examples)      │    │                              │   │
│  │  └───► Loss: 2.667409   │    │                              │   │
│  └──────────┬───────────────┘    └──────────────┬───────────────┘   │
│             ↓                                   ↓                    │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                 GROUNDED ANSWER BUILDER                     │     │
│  │  Combines Retrieval + Context + Answer Construction        │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↓                                      │
│                    ┌──────────────────┐                            │
│                    │  FLASK API       │                            │
│                    │  app.py          │                            │
│                    └──────────────────┘                            │
│                              ↓                                      │
│                    ┌──────────────────┐                            │
│                    │  WEB INTERFACE   │                            │
│                    │  index.html      │                            │
│                    └──────────────────┘                            │
│                              ↓                                      │
│                    ┌──────────────────┐                            │
│                    │  Question        │                            │
│                    │  → Answer        │                            │
│                    │  + Sources       │                            │
│                    │  + Scores        │                            │
│                    └──────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
📚 Dataset & Tokenizer
Bible Corpus
Attribute	Detail
File	data/raw/bible_corpus.txt
Format	One verse per line
Total Verses	31,102
Reference Format	Book Chapter:Verse
Sample Format:

text
Genesis 1:1 In the beginning God created the heaven and the earth.
Genesis 1:2 And the earth was without form, and void; and darkness was upon the face of the deep.
Psalm 23:1 The LORD is my shepherd; I shall not want.
John 3:16 For God so loved the world, that he gave his only begotten Son...
Custom BPE Tokenizer
Attribute	Value
Type	Byte Pair Encoding (Custom)
Vocabulary Size	2,048
BPE Merges	1,985
File	data/processed/tokenizer_v4.json
Purpose: Converts raw text into token IDs for the Transformer model.

text
Text: "In the beginning God created"
       ↓
BPE Tokenizer
       ↓
Tokens: [147, 832, 512, 104, 27, 1893, 176]
       ↓
Vocab Size: 2,048
⚙️ Transformer Model
V4 Causal Transformer Specifications
Component	Specification
Vocabulary Size	2,048
Context Length	256
Embedding Dimension	256
Attention Heads	8
Transformer Layers	6
Dropout	0.1
Total Parameters	~5.85M
Transformer Architecture
text
Input Tokens (Context Length: 256)
         ↓
Token Embeddings (Dim: 256) + Positional Embeddings
         ↓
┌─────────────────────────────────────────────────┐
│  Layer 1                                        │
│  ├── LayerNorm                                  │
│  ├── Multi-Head Attention (8 Heads)             │
│  ├── Residual Connection                        │
│  ├── LayerNorm                                  │
│  ├── Feed-Forward Network                       │
│  └── Residual Connection                        │
└─────────────────────────────────────────────────┘
         ↓ (Repeat for Layers 2-6)
         ↓
Final LayerNorm
         ↓
Linear Head (Vocab Size: 2,048)
         ↓
Output: Logits → Probabilities → Next Token Prediction
🎯 Training Configuration
Training Hyperparameters
Parameter	Value	Parameter	Value
Batch Size	2	Gradient Accumulation	16
Effective Batch Size	32	Training Steps	5,000
Initial LR	3e-4	Final LR	3e-5
Warmup Steps	300	Optimizer	AdamW
Weight Decay	0.1	Gradient Clipping	1.0
Dropout	0.1	Scheduler	Warmup + Cosine Decay
Training Results
Metric	Value
Best V4 Validation Loss	3.85433
Best QA Validation Answer Loss	2.667409
🔍 RAG System
Retrieval-Augmented Generation Pipeline
Purpose: Ground answers in actual Bible passages rather than relying solely on the Transformer's generation.

RAG Components
Component	File/Location	Details
Document Builder	src/rag/build_documents.py	Creates verse documents
Document Output	data/rag/documents.json	31,102 documents
Embedding Model	all-MiniLM-L6-v2	384-dimensional
Embedding Shape	data/rag/embeddings.npy	31,102 × 384
Retriever	src/rag/retriever.py	Semantic search
RAG Workflow
text
❓ USER QUESTION: "What does the Bible say about love?"
         │
         ▼
🔮 Question → Embedding (384-dim)
         │
         ▼
📊 Semantic Similarity Search
    ┌─────────────────────────────┐
    │  31,102 × 384 Embeddings    │
    │  Compare Against All Verses  │
    └─────────────────────────────┘
         │
         ▼
🏆 Top 20 Candidate Passages
         │
         ▼
📚 Book Diversity Selection
    ┌─────────────────────────────┐
    │  Ensure verses from          │
    │  different Bible books       │
    └─────────────────────────────┘
         │
         ▼
📖 Top 5 Passages Selected
    ┌─────────────────────────────┐
    │  1. 1 Corinthians 13:4       │
    │  2. John 3:16               │
    │  3. Psalm 23:1              │
    │  4. Romans 8:38-39          │
    │  5. 1 John 4:8              │
    └─────────────────────────────┘
         │
         ▼
✨ Grounded Answer Builder
         │
         ▼
📨 Response: Answer + Sources + Scores
🧪 QA Fine-Tuning
Experimental Question-Answering Pipeline
Purpose: Improve the Transformer's ability to answer Bible-related questions.

QA Dataset Distribution
Split	Count
Training	9,648
Validation	1,166
Testing	1,186
Total	12,000
Key Details
Aspect	Detail
Loss Masking	Answer-only (ignores question tokens)
Status	Experimental
Model Size	Small for QA tasks
Instruction Tuning	Not applied
Generation	Limited for free-form answers
Production Use	Relies on retrieval + grounded builder
📡 API Reference
Available Endpoints
Endpoint	Method	Description
/	GET	Serves the web interface
/api/ask	POST	Submit Bible questions
/health	GET	Health check endpoint
POST /api/ask Request
json
{
  "question": "What does the Bible say about love?"
}
POST /api/ask Response
json
{
  "question": "What does the Bible say about love?",
  "answer": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud...",
  "sources": [
    {
      "reference": "1 Corinthians 13:4-5",
      "text": "Love is patient, love is kind...",
      "score": 0.8234
    },
    {
      "reference": "1 John 4:8",
      "text": "Whoever does not love does not know God, because God is love.",
      "score": 0.7891
    }
  ]
}
GET /health Response
json
{
  "status": "ok"
}
🚀 Quick Start
bash
# Clone the repository
git clone https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt.git
cd ScriptureLM-SelfBuilt

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
Then open:

Web Interface: http://127.0.0.1:5000

Health Check: http://127.0.0.1:5000/health

📁 Project Structure
text
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
│       ├── embeddings.npy
│       └── qa_dataset_v4.json
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
🌐 Deployment
Render Deployment Details
Attribute	Value
Platform	Render
Plan	Free
Python Version	3.12
PyTorch	CPU-only version
WSGI Server	Gunicorn
Requirements	Pinned dependencies
Build & Start Commands
bash
# Build Command
pip install -r requirements.txt

# Start Command
gunicorn app:app
Live URLs
Service	URL
Live Application	https://scripturelm-selfbuilt.onrender.com
Health Check	https://scripturelm-selfbuilt.onrender.com/health
📝 Example Questions
Try these questions to test the system:

❓ "What does the Bible say about love?"

❓ "Who was Moses?"

❓ "What does the Bible say about faith?"

❓ "What does the Bible say about forgiveness?"

❓ "What does the Bible say about hope?"

⚠️ Limitations
Area	Limitation
Model Size	Small Transformer (~5.85M params) vs. modern LLMs
Generation	Base-model generation can be unreliable
Reasoning	QA fine-tuning has limited free-form reasoning
Retrieval Quality	Depends on embedding model and question phrasing
Grounded Output	Strictly bound to retrieved Bible passages
Hosting	Free Render plan may have cold starts
Purpose	Educational/experimental, not production-ready
💡 Note: This project prioritizes learning over performance. It demonstrates the full LLM pipeline rather than achieving state-of-the-art results.

🎯 Why I Built This
This project was built to understand the internals and workflow of LLM systems instead of simply using an existing chatbot API. Building every component from scratch provided deep insights into:

Component	What I Learned
Dataset Preparation	Handling raw text data
Tokenization	Text preprocessing and encoding
BPE	Building custom tokenizers
Embeddings	Representing text as vectors
Self-Attention	Understanding token relationships
Transformers	Deep learning architectures
Training	Optimizing model parameters
Fine-Tuning	Transfer learning for specific tasks
Semantic Search	Information retrieval techniques
RAG	Combining retrieval with generation
REST APIs	Building web services
Flask	Web application development
Deployment	Putting projects into production
👨‍💻 Author
Samuel D

🎓 B.E. Computer Science and Engineering, Prathyusha Engineering College, Graduating 2027

🐙 GitHub: https://github.com/JOSESAMUEL14

🌐 Live Demo: https://scripturelm-selfbuilt.onrender.com

📖 The Vision
"Built from Scripture.
Built to understand LLMs.
Built from scratch."

📝 License
No license has been specified for this repository.

<div align="center">
Made with ☕, 📖, and a deep curiosity for how language models truly work.

⬆ Back to Top

</div>
