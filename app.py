from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
TEMPLATE_DIR = PROJECT_DIR / "ui" / "templates"
STATIC_DIR = PROJECT_DIR / "ui" / "static"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# RAG IMPORT
# ============================================================

from rag.retriever import BibleRetriever


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)


# ============================================================
# LOAD RAG RETRIEVER
# ============================================================

print("=" * 70)
print("SCRIPTURELM WEB APPLICATION")
print("=" * 70)

print("\nLoading Bible RAG system...")

retriever = BibleRetriever()

print("Bible RAG system loaded successfully.")


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    """
    Normalize text for keyword comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_question_keywords(question):
    """
    Extract useful words from the question.
    """

    stop_words = {
        "who", "what", "where", "when", "why", "how",
        "is", "was", "were", "are",
        "the", "a", "an",
        "did", "does", "do",
        "can", "could", "would", "should",
        "about", "tell", "me",
        "according", "to",
        "say", "says",
        "bible", "scripture"
    }

    words = clean_text(question).split()

    return [
        word
        for word in words
        if word not in stop_words
    ]


def split_into_sentences(text):
    """
    Split Bible text into sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# GROUNDED ANSWER BUILDER
# ============================================================

def build_answer(question, results):
    """
    Build an answer using only retrieved Bible text.

    No external AI.
    No external API.
    No generated knowledge.
    """

    if not results:
        return "No relevant Bible passages were found."

    keywords = get_question_keywords(question)

    question_lower = question.lower()

    selected_sentences = []

    for result in results:

        text = result["text"].strip()

        sentences = split_into_sentences(text)

        for sentence in sentences:

            sentence_clean = clean_text(sentence)

            matching_keywords = sum(
                1
                for keyword in keywords
                if keyword in sentence_clean
            )

            if matching_keywords > 0:

                selected_sentences.append({
                    "reference": result["reference"],
                    "text": sentence,
                    "matches": matching_keywords,
                    "score": result["score"]
                })

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_sentences = []
    seen = set()

    for item in selected_sentences:

        key = clean_text(item["text"])

        if key not in seen:

            seen.add(key)
            unique_sentences.append(item)

    # --------------------------------------------------------
    # SORT BY RELEVANCE
    # --------------------------------------------------------

    unique_sentences.sort(
        key=lambda item: (
            item["matches"],
            item["score"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # SELECT UP TO 3 PASSAGES
    # --------------------------------------------------------

    if unique_sentences:

        selected = unique_sentences[:3]

        answer_lines = [
            item["text"]
            for item in selected
        ]

        if question_lower.startswith("who "):

            prefix = (
                "The Bible describes this person "
                "through the following retrieved passages:"
            )

        elif question_lower.startswith("what "):

            prefix = (
                "According to the retrieved "
                "Bible passages:"
            )

        elif question_lower.startswith("why "):

            prefix = (
                "The retrieved Bible passages "
                "provide the following information:"
            )

        elif question_lower.startswith("how "):

            prefix = (
                "The retrieved Bible passages "
                "describe it this way:"
            )

        else:

            prefix = (
                "Based on the retrieved "
                "Bible passages:"
            )

        return prefix + "\n\n" + " ".join(answer_lines)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    best = results[0]

    return (
        "The most relevant retrieved Bible passage is:\n\n"
        + best["text"]
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ASK API
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No request data received."
            }), 400

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "error": "Question cannot be empty."
            }), 400

        print()
        print("=" * 70)
        print("NEW QUESTION")
        print("=" * 70)
        print(question)

        # ----------------------------------------------------
        # RAG RETRIEVAL
        # ----------------------------------------------------

        results = retriever.search(
            question,
            top_k=5
        )

        # ----------------------------------------------------
        # BUILD GROUNDED ANSWER
        # ----------------------------------------------------

        answer = build_answer(
            question,
            results
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = []

        for result in results:

            sources.append({
                "reference": result["reference"],
                "text": result["text"],
                "score": result["score"]
            })

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    except Exception as error:

        print()
        print("=" * 70)
        print("API ERROR")
        print("=" * 70)
        print(error)

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SCRIPTURELM SERVER STARTING")
    print("=" * 70)

    print()
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )