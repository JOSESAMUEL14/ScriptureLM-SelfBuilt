from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DOCUMENTS_FILE = BASE_DIR / "data" / "rag" / "documents.json"
EMBEDDINGS_FILE = BASE_DIR / "data" / "rag" / "embeddings.npy"


# ============================================
# SETTINGS
# ============================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================
# MAIN
# ============================================

def main():

    print("=" * 60)
    print("SCRIPTURELM - BUILDING RAG EMBEDDINGS")
    print("=" * 60)

    # Load Bible documents
    print("\nLoading documents...")

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Total documents: {len(documents)}")

    # Load embedding model
    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded!")

    # Prepare text
    texts = []

    for doc in documents:
        text = f"{doc['reference']}: {doc['text']}"
        texts.append(text)

    # Generate embeddings
    print("\nGenerating embeddings...")
    print("This may take some time...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Convert to float32
    embeddings = embeddings.astype(np.float32)

    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings)

    print("\n" + "=" * 60)
    print("EMBEDDINGS CREATED SUCCESSFULLY!")
    print("=" * 60)

    print(f"Documents:   {len(documents)}")
    print(f"Dimensions:  {embeddings.shape[1]}")
    print(f"Shape:       {embeddings.shape}")
    print(f"Saved to:    {EMBEDDINGS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()