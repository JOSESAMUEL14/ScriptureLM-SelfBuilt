<div align="center">
✝️ SCRIPTURELM-SELFBUILT
A Self-Built Bible Language Model + Retrieval-Augmented Generation System
https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-https://scripturelm--selfbuilt.onrender.com-2ea44f?style=for-the-badge&logo=render&logoColor=white
https://img.shields.io/badge/%F0%9F%93%82_Repository-View_on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white

https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white
https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white
https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white
https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white

</div>
📜 The Sacred Blueprint
"Every line of code is a verse. Every token is a word. Every retrieval is a revelation."

ScriptureLM-SelfBuilt is an educational, domain-specific NLP project that deconstructs the modern LLM pipeline. Instead of treating AI as a black box, this project builds every component from scratch—from tokenization to transformer training, from semantic search to deployment.

🧬 The Complete Pipeline














📑 Table of Contents
📊 Project Dashboard

🧠 System Architecture

📚 Dataset & Tokenizer

⚙️ Transformer Model

🎯 Training Configuration

🔍 RAG System

🧪 QA Fine-Tuning

🚀 Application Flow

📡 API Reference

💻 Local Development

🌐 Deployment

⚠️ Limitations

🏆 Project Status

👨‍💻 Author

📊 Project Dashboard
Category	Specification	Status
📖 Corpus	31,102 Bible verses, one per line	✅
✏️ Tokenizer	Custom BPE (2,048 vocab, 1,985 merges)	✅
🧠 Model	V4 Causal Transformer	✅
🔢 Parameters	~5.85 Million	✅
🏗️ Layers	6 Transformer Layers	✅
🎯 Attention	8 Attention Heads	✅
📐 Dimensions	256 Embedding, 256 Context	✅
🔮 RAG Model	all-MiniLM-L6-v2 (384-dim)	✅
📊 QA Dataset	12,000 examples	✅
🌐 Web Server	Flask + Gunicorn	✅
☁️ Deployment	Render (Free Plan)	✅
💚 Health Check	/health endpoint	✅
🧠 System Architecture
Two-Path Design: Generation + Retrieval
ascii
╔══════════════════════════════════════════════════════════════════════════════╗
║                      SCRIPTURELM-SELFBUILT ARCHITECTURE                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   ┌──────────────────────────────┐    ┌─────────────────────────────┐     ║
║   │     LANGUAGE MODEL PATH       │    │        RAG PATH             │     ║
║   │                              │    │                             │     ║
║   │  📖 Bible Corpus            │    │  📄 Document Builder        │     ║
║   │  (31,102 verses)            │    │  └───► documents.json       │     ║
║   │         ↓                   │    │                             │     ║
║   │  ✏️ BPE Tokenizer           │    │  🔮 all-MiniLM-L6-v2       │     ║
║   │  (Vocab: 2,048)            │    │  └───► embeddings.npy      │     ║
║   │         ↓                   │    │         (31,102 × 384)     │     ║
║   │  💾 Tokenized Dataset       │    │                             │     ║
║   │  (dataset_v4.pt)           │    │  🔍 Retriever              │     ║
║   │         ↓                   │    │  └───► Top 20 Candidates   │     ║
║   │  🧠 Transformer V4         │    │                             │     ║
║   │  (6 Layers, 8 Heads)       │    │  📚 Book Diversity Select   │     ║
║   │         ↓                   │    │  └───► Top 5 Passages      │     ║
║   │  🎯 Training (5,000 steps)  │    │                             │     ║
║   │  └───► Loss: 3.85433       │    │                             │     ║
║   │         ↓                   │    │                             │     ║
║   │  🧪 QA Fine-Tuning         │    │                             │     ║
║   │  (12,000 examples)         │    │                             │     ║
║   │  └───► Loss: 2.667409      │    │                             │     ║
║   └──────────────┬───────────────┘    └─────────────┬───────────────┘     ║
║                  ↓                                  ↓                      ║
║   ┌──────────────────────────────────────────────────────────────┐         ║
║   │              GROUNDED ANSWER BUILDER                          │         ║
║   │  "Blessed are the pure in heart..." ← Context Retrieval     │         ║
║   └──────────────────────────────────────────────────────────────┘         ║
║                                    ↓                                       ║
║                    ┌──────────────────────────┐                           ║
║                    │   🌐 FLASK API (app.py)   │                           ║
║                    └──────────────────────────┘                           ║
║                                    ↓                                       ║
║                    ┌──────────────────────────┐                           ║
║                    │  🖥️ WEB INTERFACE        │                           ║
║                    │  index.html              │                           ║
║                    └──────────────────────────┘                           ║
║                                    ↓                                       ║
║                    ┌──────────────────────────┐                           ║
║                    │  ❓ Question → 📖 Answer  │                           ║
║                    │  + 📚 Sources + 📊 Scores │                           ║
║                    └──────────────────────────┘                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
📚 Dataset & Tokenizer
📖 Bible Corpus
Attribute	Detail
File	data/raw/bible_corpus.txt
Format	One verse per line
Total Verses	31,102
Reference Format	Book Chapter:Verse
<details> <summary><b>📝 Sample Corpus Format</b></summary>
text
Genesis 1:1 In the beginning God created the heaven and the earth.
Genesis 1:2 And the earth was without form, and void; and darkness was upon the face of the deep.
Psalm 23:1 The LORD is my shepherd; I shall not want.
John 3:16 For God so loved the world, that he gave his only begotten Son...
</details>
✏️ Custom BPE Tokenizer
Attribute	Value
Type	Byte Pair Encoding (Custom)
Vocabulary Size	2,048
BPE Merges	1,985
File	data/processed/tokenizer_v4.json
🎯 Purpose: Converts raw text into token IDs for the Transformer model.

ascii
Text: "In the beginning God created"
       ↓
BPE Tokenizer
       ↓
Tokens: [147, 832, 512, 104, 27, 1893, 176]
       ↓
Vocab Size: 2,048
⚙️ Transformer Model
V4 Causal Transformer Specifications
ascii
╔═══════════════════════════════════════════════════════╗
║           V4 TRANSFORMER ARCHITECTURE                ║
╠═══════════════════════════════════════════════════════╣
║ ╔═══════════════════════════════════════════════════╗ ║
║ ║  Input Tokens (Context Length: 256)              ║ ║
║ ╠═══════════════════════════════════════════════════╣ ║
║ ║   ↓ Token Embeddings (Dim: 256)                 ║ ║
║ ║   ↓ Positional Embeddings                       ║ ║
║ ╠═══════════════════════════════════════════════════╣ ║
║ ║   ┌───────────────────────────────────────────┐  ║ ║
║ ║   │  Layer 1                                 │  ║ ║
║ ║   │  ├── LayerNorm                           │  ║ ║
║ ║   │  ├── Multi-Head Attention (8 Heads)      │  ║ ║
║ ║   │  ├── Residual Connection                 │  ║ ║
║ ║   │  ├── LayerNorm                           │  ║ ║
║ ║   │  ├── Feed-Forward Network                │  ║ ║
║ ║   │  └── Residual Connection                 │  ║ ║
║ ║   └───────────────────────────────────────────┘  ║ ║
║ ║   ═══════════════════════════════════════════    ║ ║
║ ║   ┌───────────────────────────────────────────┐  ║ ║
║ ║   │  Layers 2-6 (Repeat)                     │  ║ ║
║ ║   │  └── Same Structure                      │  ║ ║
║ ║   └───────────────────────────────────────────┘  ║ ║
║ ╠═══════════════════════════════════════════════════╣ ║
║ ║   ↓ Final LayerNorm                             ║ ║
║ ║   ↓ Linear Head (Vocab Size: 2,048)            ║ ║
║ ╠═══════════════════════════════════════════════════╣ ║
║ ║  Output: Logits → Probabilities                 ║ ║
║ ╚═══════════════════════════════════════════════════╝ ║
╚═══════════════════════════════════════════════════════╝
📊 Model Details Table
Component	Specification
Vocabulary Size	2,048
Context Length	256
Embedding Dimension	256
Attention Heads	8
Transformer Layers	6
Dropout	0.1
Total Parameters	~5.85M
🎯 Training Configuration
Training Hyperparameters
Parameter	Value	Parameter	Value
Batch Size	2	Gradient Accumulation	16
Effective Batch Size	32	Training Steps	5,000
Initial LR	3e-4	Final LR	3e-5
Warmup Steps	300	Optimizer	AdamW
Weight Decay	0.1	Gradient Clipping	1.0
Dropout	0.1	Scheduler	Warmup + Cosine Decay
🏆 Training Results
ascii
╔═══════════════════════════════════════════════════════╗
║              TRAINING METRICS                         ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   Best V4 Validation Loss:   ████████░░ 3.85433      ║
║                                                       ║
║   Best QA Answer Loss:       ████████░░ 2.667409     ║
║                                                       ║
║   Training Script:           src/train_v4.py          ║
║   QA Training Script:        src/rag/train_qa_v4.py  ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
🔍 RAG System
Retrieval-Augmented Generation Pipeline
Purpose: Ground answers in actual Bible passages rather than relying solely on the Transformer's generation.

📋 RAG Components
Component	File/Location	Details
Document Builder	src/rag/build_documents.py	Creates verse documents
Document Output	data/rag/documents.json	31,102 documents
Embedding Model	all-MiniLM-L6-v2	384-dimensional
Embedding Shape	data/rag/embeddings.npy	31,102 × 384
Retriever	src/rag/retriever.py	Semantic search
🔄 RAG Workflow Diagram
ascii
╔═══════════════════════════════════════════════════════════════════════╗
║                      RAG RETRIEVAL WORKFLOW                           ║
╚═══════════════════════════════════════════════════════════════════════╝

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
      │  Ensure verses from         │
      │  different Bible books      │
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

📊 QA Dataset Distribution
ascii
╔═══════════════════════════════════════════════════════════╗
║              QA DATASET (12,000 Examples)                 ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   Training:   ████████████████████████████░░  9,648      ║
║   Validation: ██████░░░░░░░░░░░░░░░░░░░░░░░░  1,166      ║
║   Testing:    ██████░░░░░░░░░░░░░░░░░░░░░░░░  1,186      ║
║                                                           ║
║   Total:               12,000                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
🎯 Key Details
Aspect	Detail
Loss Masking	Answer-only (ignores question tokens)
Status	Experimental
Model Size	Small for QA tasks
Instruction Tuning	Not applied
Generation	Limited for free-form answers
Production Use	Relies on retrieval + grounded builder
🚀 Application Flow
Complete User Journey
ascii
╔══════════════════════════════════════════════════════════════════════════╗
║                         APPLICATION FLOW                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

   👤 USER ACTIONS                           🤖 SYSTEM RESPONSES
────────────────────────────────────────────────────────────────────────────
   
   1. Types question                      🔮 Question → Embedding
      "What does the Bible say              │
       about forgiveness?"                  ▼
                                          📊 Search 31,102 verses
                                            │
                                            ▼
                                          📚 Retrieve Top 20
                                            │
                                            ▼
                                          📖 Select 5 Diverse Passages
                                            │
   2. Clicks "Ask" ▶────────────────────▶  ✨ Build Grounded Answer
                                            │
                                            ▼
                                          📨 Return JSON Response
                                            │
   3. Sees Answer                        ┌────────────────────────┐
      "Forgive 70×7..."                  │ • Answer               │
      📚 Sources                         │ • Sources + References │
      📊 Similarity Scores               │ • Similarity Scores    │
                                         └────────────────────────┘
📡 API Reference
Available Endpoints
Endpoint	Method	Description
/	GET	Serves the web interface
/api/ask	POST	Submit Bible questions
/health	GET	Health check endpoint
📤 POST /api/ask Request
json
{
  "question": "What does the Bible say about love?"
}
📥 POST /api/ask Response
json
{
  "question": "What does the Bible say about love?",
  "answer": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.",
  "sources": [
    {
      "reference": "1 Corinthians 13:4-5",
      "text": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs.",
      "score": 0.8234
    },
    {
      "reference": "1 John 4:8",
      "text": "Whoever does not love does not know God, because God is love.",
      "score": 0.7891
    }
  ]
}
✅ GET /health Response
json
{
  "status": "ok"
}
💻 Local Development
Step-by-Step Setup
bash
# 1️⃣ Clone the repository
git clone https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt.git
cd ScriptureLM-SelfBuilt

# 2️⃣ Create virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate

# macOS/Linux:
python -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the application
python app.py
🌐 Local Access
Resource	URL
Web Interface	http://127.0.0.1:5000
Health Check	http://127.0.0.1:5000/health
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
# 🔨 Build Command
pip install -r requirements.txt

# 🚀 Start Command
gunicorn app:app
🌍 Live URLs
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
Honest Technical Assessment
Area	Limitation
Model Size	Small Transformer (~5.85M params) vs. modern LLMs
Generation	Base-model generation can be unreliable
Reasoning	QA fine-tuning has limited free-form reasoning
Retrieval Quality	Depends on embedding model and question phrasing
Grounded Output	Strictly bound to retrieved Bible passages
Hosting	Free Render plan may have cold starts
Purpose	Educational/experimental, not production-ready
💡 Note: This project prioritizes learning over performance. It demonstrates the full LLM pipeline rather than achieving state-of-the-art results.

🏆 Project Status
Development Progress
ascii
╔══════════════════════════════════════════════════════════════════════════╗
║                         COMPLETED MILESTONES                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  📖 Bible Corpus              ████████████████████████░  31,102 verses  ║
║  ✏️ BPE Tokenizer             ████████████████████████░  Vocab: 2,048   ║
║  🧠 Transformer V4            ████████████████████████░  6 Layers        ║
║  🎯 V4 Training               ████████████████████████░  Loss: 3.85433  ║
║  📊 QA Dataset                ████████████████████████░  12,000 examples║
║  🧪 QA Fine-Tuning            ████████████████████████░  Loss: 2.667409 ║
║  📄 Bible Documents           ████████████████████████░  31,102 docs    ║
║  🔮 Semantic Embeddings       ████████████████████████░  31,102 × 384   ║
║  🔍 RAG Retrieval             ████████████████████████░  Top 5 Passages ║
║  ✨ Grounded Answer Builder   ████████████████████████░  Answer + Sources║
║  🌐 Flask Application         ████████████████████████░  app.py         ║
║  📡 REST API                  ████████████████████████░  /api/ask       ║
║  💚 Health Check              ████████████████████████░  /health        ║
║  ☁️ Render Deployment         ████████████████████████░  Live           ║
║  📂 GitHub Repository         ████████████████████████░  Public         ║
║  📝 Documentation             ████████████████████████░  Complete       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
Overall Progress: ✅ 16/16 (100% Complete)

👨‍💻 Author
Samuel D
🎓 B.E. Computer Science and Engineering
🏫 Prathyusha Engineering College
📅 Graduating 2027

Connect
Platform	Link
GitHub	github.com/JOSESAMUEL14
Live Demo	scripturelm-selfbuilt.onrender.com
📖 The Vision
<div align="center">
Scripture · Intelligence · Retrieval · From Scratch
"Built from Scripture.
Built to understand LLMs.
Built from scratch."

This project is a testament to the journey of learning—
Every component, every line of code, every verse is a step toward mastery.

</div>
📜 License
No license has been specified for this repository.

<div align="center">
Made with ☕, 📖, and a deep curiosity for how language models truly work.

⬆ Back to Top

</div>
