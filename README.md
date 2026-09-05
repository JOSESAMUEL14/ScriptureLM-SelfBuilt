📖 ScriptureLM-SelfBuilt
https://img.shields.io/badge/Live_Demo-https://scripturelm--selfbuilt.onrender.com-2ea44f?style=flat-square&logo=render&logoColor=white
https://img.shields.io/badge/Repository-View_on_GitHub-181717?style=flat-square&logo=github&logoColor=white
https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white
https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white
https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white
https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white

Self-built Bible Language Model + Retrieval-Augmented Generation (RAG) System — built entirely from scratch, no external generative AI APIs used.

🌐 Live Demo
Platform	URL	Status
Render (24/7)	https://scripturelm-selfbuilt.onrender.com	✅ Always Live
Health Check	https://scripturelm-selfbuilt.onrender.com/health	✅ Operational
📑 Table of Contents
About The Project

System Architecture

Dataset & Tokenizer

Transformer Model

Training Configuration

RAG System

QA Fine-Tuning

How It Works

API Reference

Quick Start

Project Structure

Deployment

Example Questions

Limitations

Author

About The Project
ScriptureLM-SelfBuilt is an educational NLP project created to understand the complete workflow of building a domain-specific language model and a retrieval-based question-answering application from the ground up.

Core Pipeline
text
Bible Corpus → Custom BPE Tokenizer → Tokenized Dataset → Causal Transformer → 
Model Training → QA Fine-Tuning → Bible Verse Embeddings → Semantic Retrieval → 
Grounded Answer Builder → Flask API → Web Interface
⚠️ Important: The deployed application does not use an external generative AI API for the final answer. The production response is built using retrieved Bible passages and a deterministic grounded answer builder.

System Architecture
Two-Path Design: Generation + Retrieval
text
┌────────────────────────────────────────────────────────────────────────────┐
│                        SCRIPTURELM-SELFBUILT ARCHITECTURE                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌────────────────────────────┐    ┌──────────────────────────────────┐   │
│  │    LANGUAGE MODEL PATH     │    │           RAG PATH               │   │
│  │                            │    │                                  │   │
│  │  📖 Bible Corpus          │    │  📄 Document Builder             │   │
│  │  (31,102 verses)          │    │  └───► documents.json            │   │
│  │         ↓                 │    │                                  │   │
│  │  ✏️ BPE Tokenizer         │    │  🔮 all-MiniLM-L6-v2            │   │
│  │  (Vocab: 2,048)           │    │  └───► embeddings.npy           │   │
│  │         ↓                 │    │         (31,102 × 384)          │   │
│  │  💾 Tokenized Dataset     │    │                                  │   │
│  │         ↓                 │    │  🔍 Retriever                   │   │
│  │  🧠 Transformer V4        │    │  └───► Top 20 Candidates        │   │
│  │  (6 Layers, 8 Heads)      │    │                                  │   │
│  │         ↓                 │    │  📚 Book Diversity Selection    │   │
│  │  🎯 Training              │    │  └───► Top 5 Passages            │   │
│  │  (5,000 steps)            │    │                                  │   │
│  │  └───► Loss: 3.85433      │    │                                  │   │
│  │         ↓                 │    │                                  │   │
│  │  🧪 QA Fine-Tuning        │    │                                  │   │
│  │  (12,000 examples)        │    │                                  │   │
│  │  └───► Loss: 2.667409     │    │                                  │   │
│  └────────────┬───────────────┘    └──────────────┬───────────────────┘   │
│               ↓                                   ↓                        │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                 GROUNDED ANSWER BUILDER                             │   │
│  │  Combines Retrieval + Context + Answer Construction                │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                               ↓                                            │
│                     ┌──────────────────┐                                  │
│                     │  🌐 FLASK API    │                                  │
│                     │  app.py          │                                  │
│                     └──────────────────┘                                  │
│                               ↓                                            │
│                     ┌──────────────────┐                                  │
│                     │  🖥️ WEB UI       │                                  │
│                     │  index.html      │                                  │
│                     └──────────────────┘                                  │
│                               ↓                                            │
│                     ┌──────────────────┐                                  │
│                     │  ❓ Question     │                                  │
│                     │  → 📖 Answer    │                                  │
│                     │  + 📚 Sources   │                                  │
│                     │  + 📊 Scores    │                                  │
│                     └──────────────────┘                                  │
└────────────────────────────────────────────────────────────────────────────┘
Dataset & Tokenizer
Bible Corpus
Attribute	Value
File	data/raw/bible_corpus.txt
Format	One verse per line with Book, Chapter, and Verse reference
Total Verses	31,102
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
Transformer Model
V4 Causal Transformer Specifications
Component	Specification
Model Version	V4
Model Type	Causal Transformer Language Model
Vocabulary Size	2,048
Context Length	256 tokens
Embedding Dimension	256
Attention Heads	8
Transformer Layers	6
Dropout	0.1
Total Parameters	~5.85M
Transformer Components
Component	Description
Token Embeddings	Maps token IDs to dense vectors
Positional Embeddings	Adds positional information
Multi-Head Self-Attention	Captures token relationships
Feed-Forward Networks	Non-linear transformations
Residual Connections	Enables deep training
Pre-Normalization	LayerNorm before each sub-layer
Layer Normalization	Normalizes hidden states
Causal Language-Model Head	Predicts next token
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
Training Configuration
Training Hyperparameters
Parameter	Value	Parameter	Value
Batch Size	2	Gradient Accumulation	16
Effective Batch Size	32	Training Steps	5,000
Initial Learning Rate	3e-4	Final Learning Rate	3e-5
Warmup Steps	300	Optimizer	AdamW
Weight Decay	0.1	Gradient Clipping	1.0
Dropout	0.1	Scheduler	Warmup + Cosine Decay
Training Results
Metric	Value
Best V4 Validation Loss	3.85433
Training Script	src/train_v4.py
RAG System
Retrieval-Augmented Generation
Purpose: Retrieve relevant Bible passages for user questions and use those passages to build grounded responses.

RAG Components
Component	File/Location	Description
Document Builder	src/rag/build_documents.py	Converts Bible verses into retrieval documents
Document Output	data/rag/documents.json	31,102 documents
Embedding Model	all-MiniLM-L6-v2	384-dimensional semantic embeddings
Embedding Shape	data/rag/embeddings.npy	31,102 × 384
Retriever	src/rag/retriever.py	Semantic search implementation
Retrieval Process
text
1. User submits a question
         ↓
2. Question is converted into an embedding
         ↓
3. Embedding is normalized
         ↓
4. Similarity calculated against Bible verse embeddings
         ↓
5. Candidate passages are retrieved
         ↓
6. Results diversified across different Bible books
         ↓
7. Top 5 passages are selected
         ↓
8. Relevant sentences are identified
         ↓
9. Grounded answer builder constructs the response
RAG Workflow Diagram
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
QA Fine-Tuning
Experimental Question-Answering Pipeline
Purpose: Improve the Transformer's ability to answer Bible-related questions. This is an experimental component.

QA Dataset Distribution
Split	Count
Training	9,648
Validation	1,166
Testing	1,186
Total	12,000
Key Details
Aspect	Detail
Dataset	data/rag/qa_dataset_v4.json
Loss Masking	Answer-only (ignores question tokens)
Best Validation Answer Loss	2.667409
Status	Experimental
⚠️ QA Limitation: The Transformer is relatively small compared with modern large language models and is not instruction-tuned. Direct free-form QA generation is therefore limited. The deployed application relies on retrieval and grounded answer building for production responses.

How It Works
Complete User Journey
text
👤 USER ACTIONS                           🤖 SYSTEM RESPONSES
─────────────────────────────────────────────────────────────────

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
API Reference
Available Endpoints
Endpoint	Method	Description
/	GET	Web application interface
/api/ask	POST	Submit a Bible question and receive a grounded response
/health	GET	Production health check
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
      "text": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud...",
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
Quick Start
Prerequisites
Python 3.12

Git

Virtual environment (recommended)

Setup Instructions
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
Local Access
Resource	URL
Web Interface	http://127.0.0.1:5000
Health Check	http://127.0.0.1:5000/health
Project Structure
text
ScriptureLM-SelfBuilt/
│
├── app.py
├── README.md
├── requirements.txt
├── .python-version
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── bible_corpus.txt
│   │
│   ├── processed/
│   │   ├── dataset_v4.pt
│   │   └── tokenizer_v4.json
│   │
│   └── rag/
│       ├── documents.json
│       └── embeddings.npy
│
├── src/
│   ├── dataset_v4.py
│   ├── generate_v4.py
│   ├── model_v4.py
│   ├── tokenizer_v4.py
│   ├── train_v4.py
│   │
│   └── rag/
│       ├── build_documents.py
│       ├── build_embeddings.py
│       ├── build_qa_dataset_v4.py
│       ├── retriever.py
│       ├── test_rag_qa_v4.py
│       └── train_qa_v4.py
│
└── ui/
    ├── static/
    └── templates/
        └── index.html
Technology Stack
Tool	Purpose
Python	Core programming language
PyTorch	Transformer implementation and model training
NumPy	Numerical processing and embedding storage
Sentence Transformers	Semantic embedding generation
all-MiniLM-L6-v2	Pretrained embedding model for semantic retrieval
Flask	Web application backend and REST API
Gunicorn	Production WSGI server
HTML / CSS / JavaScript	Frontend
Git	Version control
GitHub	Source-code repository
Render	Cloud deployment
Deployment
Render Deployment Details
Attribute	Value
Platform	Render
Plan	Free
Python Version	3.12
PyTorch	CPU-only
Production Server	Gunicorn
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
Example Questions
Try these questions to test the system:

❓ "What does the Bible say about love?"

❓ "Who was Moses?"

❓ "What does the Bible say about faith?"

❓ "What does the Bible say about forgiveness?"

❓ "What does the Bible say about hope?"

Testing
Aspect	Detail
RAG Testing Script	src/rag/test_rag_qa_v4.py
Tested Topics	Love, Moses
Best V4 Validation Loss	3.85433
Best QA Validation Answer Loss	2.667409
Limitations
Area	Limitation
Model Size	Small Transformer compared with modern large language models
Generation	Base-model text generation can be unreliable
Reasoning	QA fine-tuned model has limited free-form reasoning capability
Retrieval Quality	Depends on embedding model and question wording
Grounded Output	Production answer system is grounded in retrieved Bible passages
Hosting	Free Render hosting can experience cold starts after inactivity
Purpose	Educational and experimental language-model and Bible retrieval system
💡 Note: This project prioritizes learning over performance. It demonstrates the full LLM pipeline rather than achieving state-of-the-art results.

Learning Areas
This project provided hands-on experience with:

Area	Component
Natural Language Processing	Dataset preparation, Tokenization, BPE
Deep Learning	Embeddings, Self-attention, Transformers
Model Development	Causal language modeling, Training, Fine-tuning
Information Retrieval	Semantic search, Retrieval-Augmented Generation
Web Development	REST APIs, Flask, Frontend
DevOps	Git, GitHub, Cloud deployment, Health checks
Project Status
Component	Status
Bible Corpus Preparation	✅ Completed
Custom BPE Tokenizer	✅ Completed
Transformer V4 Implementation	✅ Completed
V4 Model Training	✅ Completed
QA Dataset Creation	✅ Completed
QA Fine-Tuning	✅ Completed
Bible Document Generation	✅ Completed
Semantic Embeddings	✅ Completed
RAG Retrieval	✅ Completed
Grounded Answer Builder	✅ Completed
Flask Application	✅ Completed
REST API	✅ Completed
Production Health Check	✅ Completed
Render Deployment	✅ Completed
GitHub Repository	✅ Completed
Author
Samuel D

🎓 B.E. Computer Science and Engineering, Prathyusha Engineering College, Graduating 2027

🐙 GitHub: https://github.com/JOSESAMUEL14

🌐 Live Demo: https://scripturelm-selfbuilt.onrender.com

The Vision
"Built from Scripture.
Built to understand LLMs.
Built from scratch."

License
No license has been specified for this repository.

<div align="center">
Made with ☕, 📖, and a deep curiosity for how language models truly work.

⬆ Back to Top

</div>
