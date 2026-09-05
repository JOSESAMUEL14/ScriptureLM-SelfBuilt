# ScriptureLM — Self-Built Bible Language Model

<p align="center">
  <b>A self-built Language Model + RAG system for exploring the Bible.</b>
</p>

<p align="center">
  <a href="https://scripturelm-selfbuilt.onrender.com">
    🌐 <b>LIVE DEMO</b>
  </a>
  &nbsp; • &nbsp;
  <a href="https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt">
    💻 <b>GITHUB</b>
  </a>
</p>

---

## 🌐 Try ScriptureLM

> **Ask a question. Retrieve relevant Scripture. Get a grounded response.**

### 🚀 Live Application

**https://scripturelm-selfbuilt.onrender.com**

### 💻 Source Code

**https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt**

---

## 🧠 What is ScriptureLM?

**ScriptureLM-SelfBuilt** is an experimental Bible-focused language-model project built to understand how modern NLP systems work from the ground up.

Instead of simply connecting an existing chatbot API, this project explores the complete pipeline:

```text
Bible Corpus
     ↓
Custom BPE Tokenizer
     ↓
Tokenized Dataset
     ↓
Transformer Language Model
     ↓
Model Training
     ↓
QA Fine-Tuning
     ↓
Semantic Embeddings
     ↓
RAG Retrieval
     ↓
Grounded Answer Builder
     ↓
Flask Web Application

The deployed application uses retrieval-grounded Bible passages to construct its answers.

🚫 No external generative AI API is used to generate the final answer.

✨ Project Highlights
Component	Implementation
📚 Bible Corpus	31,102 verses
🔤 Tokenizer	Custom BPE
🧠 Language Model	Self-built Causal Transformer
🧩 Vocabulary	2,048 tokens
🏗️ Transformer Layers	6
👁️ Attention Heads	8
📐 Embedding Size	256
🧮 Model Size	~5.85M parameters
🔎 RAG Embeddings	384 dimensions
🤖 Embedding Model	all-MiniLM-L6-v2
🌐 Backend	Flask
🚀 Deployment	Render
💰 Hosting Cost	$0
🏗️ How It Works
1️⃣ Build the Language Model

The Bible corpus is converted into tokens using a custom BPE tokenizer.

Bible Text
    ↓
BPE Tokenizer
    ↓
Token IDs
    ↓
Training Blocks
    ↓
Transformer

The V4 Transformer contains:

6 Transformer layers
8 attention heads
256-dimensional embeddings
256-token context window
Causal self-attention
Feed-forward networks
Residual connections
Layer normalization
Dropout
🔎 2️⃣ Retrieval-Augmented Generation

The deployed application uses a separate retrieval pipeline to ground responses in Scripture.

User Question
      ↓
Question Embedding
      ↓
Semantic Similarity Search
      ↓
31,102 Bible Verses
      ↓
Top Relevant Passages
      ↓
Relevance Filtering
      ↓
Grounded Answer
Retrieval Engine

Each Bible verse is represented as a 384-dimensional embedding.

The system uses:

all-MiniLM-L6-v2

The embeddings are normalized and compared using similarity scoring.

The retriever:

Converts the question into an embedding.
Searches the Bible embedding matrix.
Finds relevant candidate verses.
Applies diversity across Bible books.
Selects the top 5 passages.
Passes them to the grounded answer builder.
📝 3️⃣ QA Fine-Tuning

The project also includes an experimental QA fine-tuning pipeline.

Dataset
12,000 QA examples

Split:

Training     → 9,648
Validation   → 1,166
Testing      → 1,186

The training process uses answer-only loss masking, allowing the model to focus its learning on the answer portion.

Best recorded validation answer loss:

2.667409
Why is QA marked experimental?

The Transformer is intentionally small compared with modern large language models and is not instruction-tuned like ChatGPT-style models.

Therefore, the production application relies on retrieval + deterministic grounding rather than depending on free-form generated answers.

🧩 System Architecture
                         SCRIPTURELM
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
   LANGUAGE MODEL PATH                     RAG PATH
          │                                       │
          ▼                                       ▼
   Bible Corpus                            Bible Verses
          │                                       │
          ▼                                       ▼
    BPE Tokenizer                         Documents
          │                                       │
          ▼                                       ▼
   Tokenized Dataset                  Sentence Embeddings
          │                                       │
          ▼                                       ▼
   Transformer V4                    Semantic Retrieval
          │                                       │
          ▼                                       ▼
      Training                            Top 5 Passages
          │                                       │
          └───────────────────┬───────────────────┘
                              │
                              ▼
                     Grounded Answer
                              │
                              ▼
                        Flask API
                              │
                              ▼
                       Web Interface
🧠 Transformer Configuration
Parameter	V4
Vocabulary Size	2,048
Context Length	256
Embedding Dimension	256
Attention Heads	8
Transformer Layers	6
Dropout	0.1
Parameters	~5.85M
Training Configuration
Batch Size             : 2
Gradient Accumulation  : 16
Effective Batch Size   : 32
Training Steps         : 5,000
Initial Learning Rate  : 3e-4
Final Learning Rate    : 3e-5
Warmup Steps           : 300
Optimizer              : AdamW
Weight Decay           : 0.1
Gradient Clipping      : 1.0
Scheduler              : Warmup + Cosine Decay

Best recorded V4 validation loss:

3.85433
📚 Dataset

The Bible corpus is stored at:

data/raw/bible_corpus.txt
Corpus Statistics
Total verses : 31,102
Format       : One verse per line
References   : Book / Chapter / Verse

Example:

[Book Chapter:Verse] verse text
🔤 Custom BPE Tokenizer

ScriptureLM uses a custom Byte Pair Encoding tokenizer.

Vocabulary Size : 2,048
BPE Merges      : 1,985

The tokenizer converts raw Bible text into token IDs that can be consumed by the Transformer.

🌐 Web Application

The web application is built using Flask.

API Endpoints
Endpoint	Method	Description
/	GET	ScriptureLM web interface
/api/ask	POST	Ask a Bible question
/health	GET	Production health check
Example Request
{
  "question": "What does the Bible say about love?"
}
Example Response
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
🗂️ Project Structure
ScriptureLM-SelfBuilt/
│
├── app.py
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
⚙️ Training Configuration

The V4 language model was trained with:

Batch size: 2
Gradient accumulation: 16
Effective batch size: 32
Training steps: 5,000
Learning rate: 3e-4 → 3e-5
Warmup steps: 300
Optimizer: AdamW
Weight decay: 0.1
Gradient clipping: 1.0
Dropout: 0.1
Learning-rate schedule: Warmup + Cosine Decay

Best recorded V4 validation loss:

3.85433
🚀 Run Locally
1. Clone the Repository
git clone https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt.git
cd ScriptureLM-SelfBuilt
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Start ScriptureLM
python app.py

Open the application:

http://127.0.0.1:5000

Health check:

http://127.0.0.1:5000/health
☁️ Production Deployment

ScriptureLM is deployed using Render's free web service.

🌐 Production Application
https://scripturelm-selfbuilt.onrender.com
❤️ Production Health Check
https://scripturelm-selfbuilt.onrender.com/health
Build Command
pip install -r requirements.txt
Start Command
gunicorn app:app

The project uses:

Python 3.12
CPU-only PyTorch
Gunicorn
Flask

This keeps the deployment suitable for free CPU hosting.

💬 Try Asking ScriptureLM

Some example questions:

What does the Bible say about love?
Who was Moses?
What does the Bible say about faith?
What does the Bible say about forgiveness?

The system retrieves relevant Scripture passages and builds a grounded response from them.

⚠️ Limitations

ScriptureLM is primarily an educational and experimental NLP project.

Current limitations include:

The Transformer is small compared with modern LLMs.
Base-model text generation can be unreliable.
QA fine-tuning has limited free-form reasoning capability.
The production answer system depends heavily on retrieval quality.
Semantic retrieval depends on the embedding model and wording of the query.
The free Render deployment can experience cold starts after inactivity.

The system should be treated as a Bible retrieval and language-model experiment, not as a replacement for authoritative Bible study resources or theological scholarship.

🎯 Why I Built This

The main goal of ScriptureLM was not simply to build a Bible chatbot.

It was to understand how an LLM-based system is actually constructed.

Through this project, I worked with:

NLP
 ↓
Tokenization
 ↓
BPE
 ↓
Embeddings
 ↓
Self-Attention
 ↓
Transformers
 ↓
Language Model Training
 ↓
Fine-Tuning
 ↓
Semantic Search
 ↓
RAG
 ↓
REST API
 ↓
Flask
 ↓
Cloud Deployment
🛠️ Technology Stack
Technology	Purpose
Python	Core development
PyTorch	Transformer model and training
NumPy	Numerical processing
Sentence Transformers	Semantic embeddings
all-MiniLM-L6-v2	Retrieval embeddings
Flask	Web application and REST API
Gunicorn	Production WSGI server
HTML/CSS/JavaScript	Frontend
Git	Version control
GitHub	Source repository
Render	Free cloud deployment
📊 Project Status
🟢 Completed
 Bible corpus preparation
 Custom BPE tokenizer
 Transformer V4 implementation
 V4 model training
 QA dataset creation
 QA fine-tuning
 Bible document generation
 Semantic embeddings
 RAG retrieval
 Grounded answer generation
 Flask API
 Web interface
 Production deployment
 Production health check
 GitHub repository
 Professional documentation
👨‍💻 Author
Samuel D

B.E. Computer Science and Engineering
Prathyusha Engineering College
Graduating 2027

GitHub

https://github.com/JOSESAMUEL14

<p align="center">
✝️ Built from Scripture.
🧠 Built to understand LLMs.
💻 Built from scratch.
</p>
📄 License

No license has been specified for this repository.


**This is the version I recommend using.** It is ready to paste directly into `README.md` wi
