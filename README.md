# 🧠 ScriptureLM-SelfBuilt

### BUILD · TRAIN · RETRIEVE · UNDERSTAND

<p align="center">
  <b>A Self-Built Bible Language Model + Retrieval-Augmented Generation System</b>
</p>

<p align="center">
  PyTorch • Custom BPE • Transformer • RAG • Flask • Render
</p>

<p align="center">
  <a href="https://scripturelm-selfbuilt.onrender.com">
    🌐 <b>LIVE DEMO</b>
  </a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt">
    💻 <b>GITHUB</b>
  </a>
</p>

---

## 📚 Table of Contents

* [About](#-about)
* [Key Highlights](#-key-highlights)
* [System Overview](#-system-overview)
* [Dataset](#-dataset)
* [Tokenizer](#-tokenizer)
* [Transformer Model](#-transformer-model)
* [Training](#-training)
* [RAG System](#-rag-system)
* [QA Fine-Tuning](#-qa-fine-tuning)
* [How It Works](#-how-it-works)
* [Web Application](#-web-application)
* [API](#-api)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Run Locally](#-run-locally)
* [Deployment](#-deployment)
* [Example Questions](#-example-questions)
* [Limitations](#-limitations)
* [Project Status](#-project-status)
* [Author](#-author)

---

## 📖 About

**ScriptureLM-SelfBuilt** is an educational NLP project created to understand how a domain-specific language model and RAG application can be built from the ground up.

The project combines a **custom-trained causal Transformer language model** with a **semantic Bible retrieval system** and a **Flask web application**.

The overall pipeline is:

```text
Bible Corpus
      ↓
Custom BPE Tokenizer
      ↓
Tokenized Dataset
      ↓
Causal Transformer
      ↓
Model Training
      ↓
QA Fine-Tuning
      ↓
Bible Verse Embeddings
      ↓
Semantic Retrieval
      ↓
Grounded Answer Builder
      ↓
Flask API
      ↓
Web Interface

The deployed application retrieves relevant Bible passages and builds the final response from those retrieved passages.

🔒 No external generative AI API is used for the final answer.

⭐ Key Highlights
Feature	Description
📚 Bible Corpus	31,102 Bible verses
🔤 Tokenizer	Custom BPE tokenizer
🧠 Language Model	Self-built causal Transformer
🏗️ Transformer Version	V4
🧮 Model Size	~5.85M parameters
👁️ Attention Heads	8
🧱 Transformer Layers	6
📐 Embedding Dimension	256
🔎 RAG Embeddings	384 dimensions
🤖 Embedding Model	all-MiniLM-L6-v2
📝 QA Dataset	12,000 examples
🌐 Backend	Flask
🔌 API	REST API
❤️ Health Check	/health
☁️ Deployment	Render
💰 Hosting	Free
🏗️ System Overview

ScriptureLM consists of two major pipelines:

                         SCRIPTURELM
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      LANGUAGE MODEL                       RAG SYSTEM
             │                                 │
             ▼                                 ▼
      Bible Corpus                         Bible Verses
             │                                 │
             ▼                                 ▼
      BPE Tokenizer                       Documents
             │                                 │
             ▼                                 ▼
      Tokenized Dataset              Sentence Embeddings
             │                                 │
             ▼                                 ▼
      Transformer V4                  Semantic Retrieval
             │                                 │
             ▼                                 ▼
         Training                       Top 5 Passages
             │                                 │
             └────────────────┬────────────────┘
                              │
                              ▼
                    Grounded Answer Builder
                              │
                              ▼
                         Flask API
                              │
                              ▼
                       Web Interface

🧠 Language Model Path: Learn how a Transformer language model is trained on a domain-specific corpus.

🔎 RAG Path: Retrieve relevant Scripture passages and use them to ground the application's responses.

📚 Dataset

The core Bible corpus is stored in:

data/raw/bible_corpus.txt
Corpus Statistics
Property	Value
Total verses	31,102
Format	One verse per line
Reference	Book / Chapter / Verse
Purpose	Transformer training + RAG

Example:

[Book Chapter:Verse] verse text

The corpus is processed into machine-readable training data before being passed to the Transformer.

🔤 Tokenizer

ScriptureLM uses a custom Byte Pair Encoding (BPE) tokenizer.

Configuration
Setting	Value
Vocabulary Size	2,048
BPE Merges	1,985

The tokenizer converts Bible text into token IDs that can be processed by the language model.

Raw Bible Text
      ↓
BPE Tokenizer
      ↓
Token IDs
      ↓
Training Sequences

Tokenizer files are stored in:

data/processed/tokenizer_v4.json
🧠 Transformer Model

The V4 model is a causal Transformer language model designed specifically for the project Bible corpus.

Model Specifications
Specification	V4
Vocabulary Size	2,048
Context Length	256
Embedding Dimension	256
Attention Heads	8
Transformer Layers	6
Dropout	0.1
Parameters	~5.85M
Model Components

The Transformer contains:

Token embeddings
Positional embeddings
Multi-head self-attention
Feed-forward networks
Residual connections
Pre-normalization
Layer normalization
Causal language-model head
Transformer Flow
Input Tokens
      ↓
Token Embeddings
      +
Positional Embeddings
      ↓
Transformer Block
      ↓
Multi-Head Self-Attention
      ↓
Feed-Forward Network
      ↓
Residual Connections
      ↓
Repeated Across 6 Layers
      ↓
Final LayerNorm
      ↓
Language Model Head
      ↓
Next Token Prediction
⚙️ Training

The V4 Transformer was trained on the processed Bible dataset.

Training Configuration
Setting	Value
Batch Size	2
Gradient Accumulation	16
Effective Batch Size	32
Training Steps	5,000
Initial Learning Rate	3e-4
Final Learning Rate	3e-5
Warmup Steps	300
Optimizer	AdamW
Weight Decay	0.1
Gradient Clipping	1.0
Dropout	0.1
Scheduler	Warmup + Cosine Decay
Training Result

Best recorded V4 validation loss:

3.85433

The training pipeline is implemented in:

src/train_v4.py
🔎 RAG System

The production application uses Retrieval-Augmented Generation (RAG) to ground responses in Bible passages.

RAG Pipeline
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
31,102 Bible Verse Embeddings
      ↓
Candidate Passages
      ↓
Book Diversity Selection
      ↓
Top 5 Relevant Passages
      ↓
Grounded Answer Builder
      ↓
Final Response
📦 Document Store

Bible verses are converted into individual documents using:

src/rag/build_documents.py

Output:

data/rag/documents.json
🧮 Embeddings

The project uses:

all-MiniLM-L6-v2

Each Bible verse is represented by a:

384-dimensional embedding

The complete embedding matrix contains:

31,102 × 384

Stored in:

data/rag/embeddings.npy
🔍 Retrieval Process

When a user asks a question:

The question is converted into an embedding.
The embedding is normalized.
Similarity is calculated against the Bible embeddings.
Relevant candidate verses are identified.
Results are diversified across Bible books.
The top 5 passages are selected.
Relevant sentences are extracted.
The grounded response is returned.

The retriever is implemented in:

src/rag/retriever.py
📝 QA Fine-Tuning

The project also includes an experimental QA fine-tuning pipeline.

Dataset:

data/rag/qa_dataset_v4.json
Dataset Split
Split	Examples
Training	9,648
Validation	1,166
Testing	1,186
Total	12,000

The QA training process uses answer-only loss masking so that the model focuses on learning the answer portion of each training example.

Best Validation Result
Answer Validation Loss: 2.667409
QA Pipeline
Question
   +
Retrieved Context
   ↓
QA Training Format
   ↓
Answer-Only Loss Masking
   ↓
Fine-Tuned V4 Model

⚠️ QA is an experimental component.
Because the Transformer is relatively small and is not instruction-tuned, direct free-form QA generation is limited. The deployed application therefore relies on the retrieval and grounded-answer pipeline.

🔄 How It Works

ScriptureLM follows this process when a user asks a question.

Step 1 — User Question
"What does the Bible say about love?"
Step 2 — Semantic Representation

The question is converted into a numerical embedding using:

all-MiniLM-L6-v2
Step 3 — Bible Retrieval

The embedding is compared with the stored Bible verse embeddings.

Question
   ↓
Embedding
   ↓
Similarity Search
   ↓
Relevant Bible Verses
Step 4 — Relevance Selection

The system selects the most relevant passages while applying book diversity.

Step 5 — Grounded Answer

Relevant sentences from the retrieved passages are selected and combined into the final response.

Step 6 — Web Response

The Flask API returns:

Answer
+
Source Passages
+
Similarity Scores
🌐 Web Application

The web application is built using Flask.

Main application:

app.py
Application Flow
Browser
   ↓
Flask Web Interface
   ↓
/api/ask
   ↓
BibleRetriever
   ↓
Semantic Search
   ↓
Grounded Answer Builder
   ↓
JSON Response
   ↓
Browser
🔌 API
Available Endpoints
Endpoint	Method	Purpose
/	GET	Web application
/api/ask	POST	Ask a Bible question
/health	GET	Production health check
💬 Ask Question

Endpoint:

POST /api/ask

Example request:

{
  "question": "What does the Bible say about love?"
}

Example response:

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
❤️ Health Check

Endpoint:

GET /health

Response:

{
  "status": "ok"
}

Production health check:

https://scripturelm-selfbuilt.onrender.com/health
📁 Project Structure
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
🛠️ Technology Stack
Technology	Purpose
🐍 Python	Core development
🔥 PyTorch	Transformer implementation and training
🔢 NumPy	Numerical processing
🤗 Sentence Transformers	Semantic embeddings
🔎 all-MiniLM-L6-v2	Bible verse retrieval
🌐 Flask	Web application + REST API
⚡ Gunicorn	Production WSGI server
🎨 HTML/CSS/JavaScript	Frontend
📦 Git	Version control
🐙 GitHub	Source repository
☁️ Render	Cloud deployment
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
4. Start the Application
python app.py

The application will be available at:

http://127.0.0.1:5000

Health check:

http://127.0.0.1:5000/health
☁️ Deployment

ScriptureLM is deployed using Render's free web service.

🌐 Live Application
https://scripturelm-selfbuilt.onrender.com
❤️ Health Check
https://scripturelm-selfbuilt.onrender.com/health
Build Command
pip install -r requirements.txt
Start Command
gunicorn app:app
Runtime
Python 3.12
CPU-only PyTorch
Gunicorn
Flask

The production dependencies are pinned in:

requirements.txt

This helps keep the deployment environment predictable.

💬 Example Questions

Try ScriptureLM with questions such as:

What does the Bible say about love?
Who was Moses?
What does the Bible say about faith?
What does the Bible say about forgiveness?
What does the Bible say about hope?

The application retrieves relevant Bible passages and constructs a grounded response from those passages.

🧪 Validation & Testing

The project includes dedicated RAG testing:

src/rag/test_rag_qa_v4.py

Example retrieval topics tested during development include:

Love
Moses
Faith

The retrieval system successfully identifies relevant Bible passages from the 31,102-verse corpus.

⚠️ Limitations

ScriptureLM is an educational and experimental NLP project.

Current limitations
The Transformer is small compared with modern LLMs.
Base-model text generation can be unreliable.
The QA fine-tuned model has limited free-form reasoning ability.
Production responses depend on retrieval quality.
Retrieval quality depends on the embedding model and question wording.
The application does not perform broad theological reasoning beyond its retrieved source passages.
Free Render hosting can introduce cold-start delays after inactivity.

ScriptureLM should be considered a language-model and Bible retrieval experiment, not a replacement for authoritative Bible study resources or theological scholarship.

🎯 What This Project Demonstrates

The main purpose of this project was to understand the components behind modern LLM applications rather than simply consume an existing chatbot API.

The project covers:

📚 Dataset Preparation
       ↓
🔤 Tokenization
       ↓
🧩 BPE
       ↓
🧠 Transformer Architecture
       ↓
👁️ Self-Attention
       ↓
⚙️ Model Training
       ↓
📝 Fine-Tuning
       ↓
📐 Embeddings
       ↓
🔎 Semantic Search
       ↓
📦 RAG
       ↓
🔌 REST API
       ↓
🌐 Flask
       ↓
☁️ Cloud Deployment
📊 Project Status
🟢 COMPLETED & DEPLOYED
Component	Status
Bible Corpus	✅ Completed
BPE Tokenizer	✅ Completed
Transformer V4	✅ Completed
V4 Training	✅ Completed
QA Dataset	✅ Completed
QA Fine-Tuning	✅ Completed
Bible Documents	✅ Completed
Semantic Embeddings	✅ Completed
RAG Retrieval	✅ Completed
Grounded Answer Builder	✅ Completed
Flask Application	✅ Completed
REST API	✅ Completed
Health Check	✅ Completed
Render Deployment	✅ Completed
GitHub Repository	✅ Completed
Documentation	✅ Completed
🌐 Live Project
🚀 ScriptureLM

https://scripturelm-selfbuilt.onrender.com

💻 Source Code

https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt

❤️ Health Check

https://scripturelm-selfbuilt.onrender.com/health

👨‍💻 Author
Samuel D

B.E. Computer Science and Engineering
Prathyusha Engineering College
Graduating 2027

GitHub

https://github.com/JOSESAMUEL14

<p align="center">
✝️ Scripture · 🧠 Intelligence · 🔎 Retrieval · 💻 From Scratch

Built from Scripture.
Built to understand LLMs.
Built from scratch.

</p>
📄 License

No license has been specified for this repository.
