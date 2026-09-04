import sys
import json
from pathlib import Path

import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.model_v4 import LanguageModel
from src.rag.retriever import BibleRetriever


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "scripturelm_v4_qa.pth"
)

TOKENIZER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "tokenizer_v4.json"
)


# ============================================================
# SETTINGS
# ============================================================

BLOCK_SIZE = 256
TOP_K = 5
ANSWER_PASSAGES = 3


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SCRIPTURELM V4 RAG + QA TEST")
print("=" * 70)

print(f"Device: {device}")
print(f"QA Model: {MODEL_PATH}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8"
) as f:
    tokenizer_data = json.load(f)


tokens = tokenizer_data["tokens"]
merges = tokenizer_data["merges"]
vocab_size = tokenizer_data["vocab_size"]


token_to_id = {
    token: i
    for i, token in enumerate(tokens)
}


id_to_token = {
    i: token
    for i, token in enumerate(tokens)
}


# ============================================================
# PREPARE MERGE RANKS
# ============================================================

merge_ranks = {}

for rank, merge in enumerate(merges):

    if isinstance(merge, list):

        if len(merge) == 2:

            pair = (
                merge[0],
                merge[1]
            )

            merge_ranks[pair] = rank

    elif isinstance(merge, str):

        parts = merge.split()

        if len(parts) == 2:

            pair = (
                parts[0],
                parts[1]
            )

            merge_ranks[pair] = rank


# ============================================================
# BPE MERGE
# ============================================================

def merge_pair(tokens_list, pair):

    result = []

    i = 0

    while i < len(tokens_list):

        if (
            i < len(tokens_list) - 1
            and tokens_list[i] == pair[0]
            and tokens_list[i + 1] == pair[1]
        ):

            result.append(
                tokens_list[i] + tokens_list[i + 1]
            )

            i += 2

        else:

            result.append(
                tokens_list[i]
            )

            i += 1

    return result


# ============================================================
# ENCODE
# ============================================================

def encode(text):

    chars = list(text)

    while True:

        pairs = [
            (chars[i], chars[i + 1])
            for i in range(len(chars) - 1)
        ]

        valid_pairs = [
            pair
            for pair in pairs
            if pair in merge_ranks
        ]

        if not valid_pairs:
            break

        best_pair = min(
            valid_pairs,
            key=lambda pair: merge_ranks[pair]
        )

        chars = merge_pair(
            chars,
            best_pair
        )

    token_ids = []

    for token in chars:

        token_ids.append(
            token_to_id.get(token, 0)
        )

    return token_ids


# ============================================================
# DECODE
# ============================================================

def decode(ids):

    output = []

    for token_id in ids:

        if token_id in id_to_token:

            output.append(
                id_to_token[token_id]
            )

    return "".join(output)


# ============================================================
# TEST TOKENIZER
# ============================================================

print("\nTesting tokenizer...")

test_text = "What does Scripture teach about love?"

test_ids = encode(test_text)

test_decoded = decode(test_ids)

print(f"Original : {test_text}")
print(f"Token count: {len(test_ids)}")
print(f"Token IDs: {test_ids[:20]}")
print(f"Decoded  : {test_decoded}")

if test_decoded == test_text:

    print("Tokenizer round-trip: PASSED")

else:

    print("Tokenizer round-trip: FAILED")


# ============================================================
# LOAD QA MODEL
# ============================================================

print("\nLoading QA model...")

model = LanguageModel()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(device)
model.eval()

print("QA model loaded successfully!")


# ============================================================
# LOAD RAG RETRIEVER
# ============================================================

print("\nLoading RAG retriever...")

retriever = BibleRetriever()

print("RAG retriever loaded successfully!")

print("=" * 70)


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(question):

    results = retriever.search(
        question,
        top_k=TOP_K
    )

    return results


# ============================================================
# BUILD GROUNDED ANSWER
# ============================================================

def generate_answer(results):

    if not results:

        return "No relevant Bible passages were found."

    selected = results[:ANSWER_PASSAGES]

    answer_parts = []

    for result in selected:

        reference = result["reference"]
        text = result["text"].strip()

        answer_parts.append(
            f"{reference}: {text}"
        )

    return "\n\n".join(answer_parts)


# ============================================================
# TEST QUESTIONS
# ============================================================

questions = [

    "What does Scripture teach about love?",

    "Who was Moses?",

    "What does the Bible say about forgiveness?",

    "What does Scripture teach about faith?"

]


# ============================================================
# RUN TEST
# ============================================================

for question in questions:

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    print("\nRetrieving Bible passages...")

    results = retrieve_context(
        question
    )

    # --------------------------------------------------------
    # DISPLAY RETRIEVED VERSES
    # --------------------------------------------------------

    print("\nRETRIEVED BIBLE PASSAGES")
    print("-" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['reference']} "
            f"(score: {result['score']:.4f})"
        )

        print(
            result["text"]
        )

        print()

    # --------------------------------------------------------
    # FINAL GROUNDED ANSWER
    # --------------------------------------------------------

    print("\nGROUNDED ANSWER")
    print("-" * 70)

    answer = generate_answer(
        results
    )

    print(answer)

    print("-" * 70)


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print("=" * 70)
print("RAG + QA TEST COMPLETE")
print("=" * 70)