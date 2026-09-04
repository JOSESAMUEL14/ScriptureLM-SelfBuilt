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

CANDIDATE_K = 20


class BibleRetriever:

    def __init__(self):

        print("Loading Bible documents...")

        with open(
            DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.documents = json.load(f)

        print(
            f"Loaded documents: "
            f"{len(self.documents)}"
        )

        print("Loading embeddings...")

        self.embeddings = np.load(
            EMBEDDINGS_FILE
        )

        print(
            f"Loaded embeddings: "
            f"{self.embeddings.shape}"
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Retriever ready!")


    # =========================================
    # SEARCH
    # =========================================

    def search(
        self,
        query,
        top_k=5
    ):

        # -------------------------------------
        # Convert question to embedding
        # -------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )


        # -------------------------------------
        # Calculate similarity
        # -------------------------------------

        scores = np.dot(
            self.embeddings,
            query_embedding
        )


        # -------------------------------------
        # Retrieve larger candidate pool
        # -------------------------------------

        candidate_k = min(
            CANDIDATE_K,
            len(self.documents)
        )

        candidate_indices = np.argsort(
            scores
        )[::-1][:candidate_k]


        # -------------------------------------
        # Select diverse results
        # -------------------------------------

        selected_indices = []

        selected_books = set()


        for index in candidate_indices:

            document = self.documents[index]

            book = document["book"]


            # Always take the highest scoring result
            if len(selected_indices) == 0:

                selected_indices.append(index)
                selected_books.add(book)

                continue


            # Prefer different books
            if book not in selected_books:

                selected_indices.append(index)
                selected_books.add(book)


            # Stop when enough results are selected
            if len(selected_indices) >= top_k:

                break


        # -------------------------------------
        # Fallback if not enough diverse books
        # -------------------------------------

        if len(selected_indices) < top_k:

            for index in candidate_indices:

                if index not in selected_indices:

                    selected_indices.append(index)

                if len(selected_indices) >= top_k:

                    break


        # -------------------------------------
        # Sort selected results by score
        # -------------------------------------

        selected_indices = sorted(
            selected_indices,
            key=lambda index: scores[index],
            reverse=True
        )


        # -------------------------------------
        # Build results
        # -------------------------------------

        results = []

        for index in selected_indices:

            document = self.documents[index]

            results.append({

                "reference":
                    document["reference"],

                "text":
                    document["text"],

                "score":
                    float(scores[index])

            })


        return results


# ============================================
# TEST
# ============================================

def main():

    print("=" * 60)
    print("SCRIPTURELM RAG RETRIEVER TEST")
    print("=" * 60)

    retriever = BibleRetriever()

    query = input(
        "\nEnter your Bible question: "
    )

    results = retriever.search(
        query,
        top_k=5
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TOP RETRIEVED VERSES"
    )

    print(
        "=" * 60
    )


    for i, result in enumerate(
        results,
        1
    ):

        print(
            f"\n{i}. "
            f"{result['reference']}"
        )

        print(
            f"   Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"   {result['text']}"
        )


    print(
        "\n" + "=" * 60
    )


if __name__ == "__main__":

    main()