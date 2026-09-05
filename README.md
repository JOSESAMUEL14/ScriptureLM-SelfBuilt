ScriptureLM-SelfBuilt

🔗 Live Demo

https://scripturelm-selfbuilt.onrender.com

📌 GitHub Repository

https://github.com/JOSESAMUEL14/ScriptureLM-SelfBuilt

📖 About the Project

ScriptureLM-SelfBuilt is a self-built Bible Language Model and Retrieval-Augmented Generation (RAG) application.

The project was built from scratch to understand the complete pipeline of a domain-specific language model:

Bible Corpus → Tokenizer → Transformer Language Model → Training → RAG → Flask API → Web Application

The application allows users to ask questions about the Bible and receive answers grounded in retrieved Bible passages.

Important: The application does not use an external generative AI API. The final web answer is built from retrieved Bible text.

✨ Features

📚 Bible corpus containing 31,102 verses

🔤 Custom BPE tokenizer

🧠 Self-built causal Transformer language model

⚙️ V4 Transformer model with approximately 5.85M parameters

📦 RAG pipeline for Bible verse retrieval

🔎 Semantic search using sentence embeddings

📊 384-dimensional normalized embeddings

📝 QA dataset and QA fine-tuning pipeline

🌐 Flask web application

🔌 REST API for Bible questions

❤️ Production health-check endpoint

🚀 Free deployment on Render

🔒 No external generative AI/API dependency

🏗️ Architecture

                    ┌─────────────────────┐
                    │   Bible Corpus      │
                    │   31,102 verses     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   BPE Tokenizer     │
                    │   Vocab: 2048       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transformer LM      │
                    │ 6 Layers            │
                    │ 8 Attention Heads   │
                    │ Embedding: 256      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    │   V4                │
                    └─────────────────────┘


User Question
      │
      ▼
┌──────────────────────┐
│ Sentence Transformer │
│ all-MiniLM-L6-v2     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Semantic Retrieval   │
│ 31,102 Bible verses  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Top Relevant         │
│ Bible Passages       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Grounded Answer      │
│ Builder              │
└──────────┬───────────┘
           │
           ▼
        User

🧠 Model Details

Transformer Language Model

The V4 model is a causal Transformer language model built specifically for the Bible corpus.

Parameter

Value

Vocabulary Size

2,048

Context / Block Size

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

The model uses:

Token embeddings

Positional embeddings

Multi-head self-attention

Feed-forward networks

Pre-normalization

Residual connections

Final LayerNorm

Causal language-model head

🔤 Tokenizer

ScriptureLM uses a custom Byte Pair Encoding (BPE) tokenizer.

Configuration:

Vocabulary size: 2,048

Number of merges: 1,985

The tokenizer converts Bible text into token IDs that can be processed by the Transformer.

📚 Dataset

The project uses a Bible corpus stored in:

data/raw/bible_corpus.txt

The corpus contains:

31,102 verses

One verse per line

Book, chapter, and verse references

Example format:

[Book Chapter:Verse] verse text

🔎 Retrieval-Augmented Generation (RAG)

The RAG system provides the factual grounding used by the web application.

Document creation

Bible verses are converted into individual documents using:

src/rag/build_documents.py

Output:

data/rag/documents.json

Embedding generation

Embeddings are generated using:

all-MiniLM-L6-v2

The resulting embedding matrix contains:

31,102 × 384

and is stored in:

data/rag/embeddings.npy

Retrieval

When a user asks a question:

The question is converted into an embedding.

The embedding is normalized.

Similarity is calculated against the Bible verse embeddings.

Candidate passages are retrieved.

Results are diversified across Bible books.

The top 5 passages are returned.

The grounded answer builder selects relevant sentences.

📝 QA Fine-Tuning

The project also includes a QA fine-tuning pipeline.

Dataset:

data/rag/qa_dataset_v4.json

The QA dataset contains:

12,000 examples

9,648 training examples

1,166 validation examples

1,186 test examples

The QA training process uses answer-only loss masking so that the model focuses its learning on the answer portion.

Best recorded validation answer loss:

2.667409

The QA model is included as an experimental component. Because the underlying language model is relatively small and is not a large instruction-tuned model, direct free-form QA generation is limited. The deployed application therefore uses the deterministic grounded answer builder with retrieved Bible passages.

🌐 Web Application

The Flask application is located in:

app.py

Main routes

Route

Method

Purpose

/

GET

Web application

/api/ask

POST

Ask a Bible question

/health

GET

Production health check

Example API request

{
  "question": "What does the Bible say about love?"
}

API response structure

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

🖥️ Project Structure

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

Learning-rate schedule: warmup + cosine decay

Best recorded V4 validation loss:

3.85433

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

The application will be available at:

http://127.0.0.1:5000

Health check:

http://127.0.0.1:5000/health

☁️ Deployment

The application is deployed as a Flask web service on Render.

Production URL:

https://scripturelm-selfbuilt.onrender.com

Production health check:

https://scripturelm-selfbuilt.onrender.com/health

The deployment uses:

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app

The project is configured for Python 3.12 and CPU-only PyTorch to remain suitable for free CPU hosting.

💬 Example Questions

You can try questions such as:

What does the Bible say about love?

Who was Moses?

What does the Bible say about faith?

The application retrieves relevant Bible passages and builds the response from those retrieved passages.

⚠️ Limitations

This project is intentionally a self-built and educational language-model project, so it has limitations.

The Transformer model is relatively small compared with modern large language models.

Generated text from the base language model is not always reliable.

The QA fine-tuned model has limited free-form reasoning ability.

The deployed answer system is primarily grounded in retrieved Bible passages.

Retrieval quality depends on the embedding model and query.

Render's free hosting can have cold starts after periods of inactivity.

The application is not intended to replace professional theological scholarship or authoritative Bible study resources.

🎯 Learning Goals

This project was developed to gain practical understanding of:

Natural Language Processing

Tokenization

BPE

Transformers

Self-attention

Causal language modeling

Model training

Fine-tuning

Embeddings

Semantic search

Retrieval-Augmented Generation

Flask

REST APIs

Web deployment

Production health checks

Git and GitHub

🛠️ Technology Stack

Technology

Purpose

Python

Core development

PyTorch

Transformer model and training

NumPy

Numerical processing

Sentence Transformers

Semantic embeddings

all-MiniLM-L6-v2

Retrieval embeddings

Flask

Web application and REST API

Gunicorn

Production WSGI server

HTML/CSS/JavaScript

Frontend

Git

Version control

GitHub

Source repository

Render

Free cloud deployment

📌 Project Status

Status: Completed and deployed ✅

Current production application:

https://scripturelm-selfbuilt.onrender.com

👨‍💻 Author

Samuel D

B.E. Computer Science and Engineering
Prathyusha Engineering College
Graduating 2027

GitHub:

https://github.com/JOSESAMUEL14

📄 License

No license has been specified for this repository.
