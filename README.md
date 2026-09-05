ScriptureLM-SelfBuilt

A self-built Bible Language Model + Retrieval-Augmented Generation (RAG) system

Live Demo · GitHub Repository

📖 Overview

ScriptureLM-SelfBuilt is an educational NLP project built to understand the complete workflow behind a domain-specific language model.

The project combines a custom-trained Transformer language model with a semantic Bible retrieval system and a Flask web application.

The complete pipeline

Bible Corpus
     ↓
BPE Tokenizer
     ↓
Tokenized Dataset
     ↓
Transformer Language Model
     ↓
Training
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

The deployed application answers questions using retrieved Bible passages.

No external generative AI API is used for the final answer.

✨ Key Features

📚 31,102 Bible verses used as the core corpus

🔤 Custom Byte Pair Encoding (BPE) tokenizer

🧠 Self-built causal Transformer language model

🏗️ V4 model with approximately 5.85M parameters

🔎 Semantic Bible verse retrieval

📐 384-dimensional sentence embeddings

📝 Experimental QA fine-tuning pipeline

🌐 Flask-based web application

🔌 REST API for Bible questions

❤️ Production health-check endpoint

🚀 Deployed on Render

💰 Hosted using the free Render plan

🏗️ System Architecture

                         SCRIPTURELM
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       LANGUAGE MODEL                       RAG SYSTEM
              │                               │
              ▼                               ▼
       Bible Corpus                      Bible Verses
              │                               │
              ▼                               ▼
       BPE Tokenizer                     Documents
              │                               │
              ▼                               ▼
      Tokenized Dataset              Sentence Embeddings
              │                               │
              ▼                               ▼
       Transformer V4                 Semantic Retrieval
              │                               │
              ▼                               ▼
           Training                     Top 5 Passages
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
                     Grounded Answer
                              │
                              ▼
                         Flask API
                              │
                              ▼
                       Web Interface

🧠 Transformer Model

The V4 model is a causal Transformer language model trained specifically on the project Bible corpus.

Model Configuration

Parameter

Value

Vocabulary Size

2,048

Context Length

256

Embedding Dimension

256

Attention Heads

8

Transformer Layers

6

Dropout

0.1

Parameters

~5.85M

Main Components

Token embeddings

Positional embeddings

Multi-head self-attention

Feed-forward networks

Residual connections

Pre-normalization

Final LayerNorm

Causal language-model head

🔤 Custom BPE Tokenizer

ScriptureLM uses a custom Byte Pair Encoding tokenizer.

Setting

Value

Vocabulary Size

2,048

BPE Merges

1,985

The tokenizer converts Bible text into token IDs that can be processed by the Transformer.

📚 Bible Corpus

The main corpus is stored in:

data/raw/bible_corpus.txt

Corpus

Property

Value

Total verses

31,102

Format

One verse per line

Reference format

Book / Chapter / Verse

Example:

[Book Chapter:Verse] verse text

🔎 Retrieval-Augmented Generation

The production application uses RAG to ground responses in Bible passages.

Retrieval Flow

User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Bible Embedding Matrix
      ↓
Candidate Passages
      ↓
Book Diversity Selection
      ↓
Top 5 Results
      ↓
Grounded Answer Builder
      ↓
Response

Embedding System

The project uses:

all-MiniLM-L6-v2

Each Bible verse is represented by a 384-dimensional embedding.

Generated embeddings are stored in:

data/rag/embeddings.npy

Bible documents are stored in:

data/rag/documents.json

The retriever normalizes embeddings and uses similarity scoring to identify relevant passages.

📝 QA Fine-Tuning

The project also contains an experimental QA fine-tuning pipeline.

Dataset:

data/rag/qa_dataset_v4.json

Dataset Split

Split

Examples

Training

9,648

Validation

1,166

Testing

1,186

Total

12,000

The training process uses answer-only loss masking, allowing the model to focus on the answer portion of each example.

Best recorded validation answer loss:

2.667409

Why is the QA system experimental?

The Transformer is intentionally small compared with modern large language models and is not instruction-tuned.

Because of this, direct free-form QA generation is limited.

The deployed application therefore relies on the retrieval + grounded answer pipeline for its responses.

⚙️ Training Configuration

The V4 language model was trained using:

Setting

Value

Batch Size

2

Gradient Accumulation

16

Effective Batch Size

32

Training Steps

5,000

Initial Learning Rate

3e-4

Final Learning Rate

3e-5

Warmup Steps

300

Optimizer

AdamW

Weight Decay

0.1

Gradient Clipping

1.0

Dropout

0.1

Scheduler

Warmup + Cosine Decay

Best recorded V4 validation loss:

3.85433

🌐 Web Application

The Flask application is located in:

app.py

API Endpoints

Endpoint

Method

Purpose

/

GET

Web interface

/api/ask

POST

Ask a Bible question

/health

GET

Production health check

Example Request

{
  "question": "What does the Bible say about love?"
}

Response

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

🚀 Run Locally

1. Clone the repository

git clone https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt.git
cd ScriptureLM-SelfBuilt

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Start the application

python app.py

Open the application:

http://127.0.0.1:5000

Health check:

http://127.0.0.1:5000/health

☁️ Deployment

The application is deployed on Render using the free web service.

Production URL

https://scripturelm-selfbuilt.onrender.com

Production Health Check

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

The dependency versions are pinned in requirements.txt for more predictable deployment.

💬 Example Questions

Try questions such as:

What does the Bible say about love?

Who was Moses?

What does the Bible say about faith?

What does the Bible say about forgiveness?

The system retrieves relevant Bible passages and constructs a grounded response from those passages.

⚠️ Limitations

ScriptureLM is an educational and experimental NLP project.

Current limitations include:

The Transformer is small compared with modern LLMs.

Base-model text generation can be unreliable.

QA fine-tuning has limited free-form reasoning capability.

Production responses depend on retrieval quality.

Retrieval quality depends on the embedding model and question wording.

The free Render service can experience cold starts after inactivity.

The project is intended as a language-model and Bible retrieval experiment, not as a replacement for authoritative Bible study resources or theological scholarship.

🎯 What This Project Demonstrates

This project was built to gain practical experience with:

Natural Language Processing

BPE tokenization

Token embeddings

Positional embeddings

Self-attention

Transformers

Causal language modeling

Model training

Fine-tuning

Sentence embeddings

Semantic search

Retrieval-Augmented Generation

Flask REST APIs

Git and GitHub

Cloud deployment

Production health checks

🛠️ Technology Stack

Technology

Role

Python

Core development

PyTorch

Transformer model and training

NumPy

Numerical processing

Sentence Transformers

Embedding generation

all-MiniLM-L6-v2

Semantic retrieval

Flask

Web application and REST API

Gunicorn

Production WSGI server

HTML / CSS / JavaScript

Frontend

Git

Version control

GitHub

Source repository

Render

Cloud deployment

📊 Project Status

🟢 Completed and Deployed

Bible corpus preparation

Custom BPE tokenizer

Transformer V4 implementation

V4 model training

QA dataset creation

QA fine-tuning

Bible document generation

Semantic embeddings

RAG retrieval

Grounded answer builder

Flask API

Web interface

Render deployment

Production health check

GitHub repository

Project documentation

👨‍💻 Author

Samuel D

B.E. Computer Science and Engineering
Prathyusha Engineering College
Graduating 2027

GitHub:
https://github.com/JOSESAMUEL14

<p align="center">
  <b>✝️ Scripture</b> · <b>🧠 Language Models</b> · <b>🔎 Retrieval</b> · <b>💻 From Scratch</b>
</p>

📄 License

No license has been specified for this repository.
