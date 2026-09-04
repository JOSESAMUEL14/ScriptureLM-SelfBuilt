import json
from collections import Counter
from pathlib import Path


# ============================================================
# SCRIPTURELM V4
# IMPROVED BPE TOKENIZER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "../data/raw/bible_corpus.txt"
OUTPUT_FILE = BASE_DIR / "../data/processed/tokenizer_v4.json"

VOCAB_SIZE = 2048


# ============================================================
# 1. LOAD CORPUS
# ============================================================

print("=" * 60)
print("ScriptureLM V4 - BPE Tokenizer")
print("=" * 60)
print()

print("Loading Bible corpus...")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

print("Characters:", len(text))
print()


# ============================================================
# 2. BASIC CLEANING
# ============================================================
#
# Keep the Bible wording.
#
# Remove verse-reference brackets such as:
#
# [Genesis 1:1]
#
# This prevents the model from learning malformed references.
#
# We keep normal spaces and newlines.
# ============================================================

import re

clean_text = re.sub(
    r"\[[^\]\n]+\]",
    "",
    text
)

# Normalize excessive spaces while preserving newlines.
clean_text = re.sub(
    r"[ \t]+",
    " ",
    clean_text
)

clean_text = re.sub(
    r"\n{3,}",
    "\n\n",
    clean_text
)

print("Original characters:", len(text))
print("Cleaned characters:", len(clean_text))
print()


# ============================================================
# 3. INITIAL VOCABULARY
# ============================================================

characters = sorted(set(clean_text))

print(
    "Initial character vocabulary:",
    len(characters)
)

print()


# ============================================================
# 4. INITIAL SYMBOL SEQUENCE
# ============================================================

symbols = list(clean_text)


# ============================================================
# 5. COUNT PAIRS
# ============================================================

def count_pairs(sequence):

    counts = Counter()

    for i in range(len(sequence) - 1):

        pair = (
            sequence[i],
            sequence[i + 1]
        )

        counts[pair] += 1

    return counts


# ============================================================
# 6. MERGE PAIR
# ============================================================

def merge_pair(sequence, pair):

    result = []

    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):

            result.append(
                pair[0] + pair[1]
            )

            i += 2

        else:

            result.append(
                sequence[i]
            )

            i += 1

    return result


# ============================================================
# 7. TRAIN BPE
# ============================================================

print("=" * 60)
print("STARTING BPE TRAINING")
print("=" * 60)
print()

print("Target vocabulary:", VOCAB_SIZE)
print()

token_set = set(characters)

merge_rules = []

current_sequence = symbols

iteration = 0


while len(token_set) < VOCAB_SIZE:

    iteration += 1

    pair_counts = count_pairs(
        current_sequence
    )

    if not pair_counts:
        break

    best_pair, best_count = (
        pair_counts.most_common(1)[0]
    )

    if best_count < 2:
        break

    merged_token = (
        best_pair[0]
        +
        best_pair[1]
    )

    # --------------------------------------------------------
    # Find another pair if the resulting token already exists.
    # --------------------------------------------------------

    if merged_token in token_set:

        found_pair = False

        for pair, count in pair_counts.most_common():

            candidate = (
                pair[0]
                +
                pair[1]
            )

            if (
                candidate not in token_set
                and count >= 2
            ):

                best_pair = pair
                best_count = count
                merged_token = candidate

                found_pair = True

                break

        if not found_pair:
            break

    # --------------------------------------------------------
    # Perform merge
    # --------------------------------------------------------

    current_sequence = merge_pair(
        current_sequence,
        best_pair
    )

    merge_rules.append(
        [
            best_pair[0],
            best_pair[1]
        ]
    )

    token_set.add(
        merged_token
    )

    if (
        iteration % 50 == 0
        or len(token_set) >= VOCAB_SIZE
    ):

        print(
            "Iteration:",
            iteration,
            "| Vocabulary:",
            len(token_set),
            "| Pair frequency:",
            best_count
        )


print()
print("=" * 60)
print("BPE TRAINING COMPLETE")
print("=" * 60)
print()

print(
    "Final vocabulary:",
    len(token_set)
)

print(
    "Total merges:",
    len(merge_rules)
)

print()


# ============================================================
# 8. CREATE TOKEN IDs
# ============================================================

tokens = sorted(
    token_set
)

token_to_id = {
    token: index
    for index, token
    in enumerate(tokens)
}

id_to_token = {
    index: token
    for token, index
    in token_to_id.items()
}


# ============================================================
# 9. ENCODE
# ============================================================

def encode_text(input_text):

    sequence = list(input_text)

    for pair in merge_rules:

        sequence = merge_pair(
            sequence,
            (
                pair[0],
                pair[1]
            )
        )

    token_ids = []

    for token in sequence:

        if token not in token_to_id:

            raise ValueError(
                "Unknown token: "
                + repr(token)
            )

        token_ids.append(
            token_to_id[token]
        )

    return token_ids


# ============================================================
# 10. DECODE
# ============================================================

def decode(token_ids):

    result = []

    for token_id in token_ids:

        result.append(
            id_to_token[
                int(token_id)
            ]
        )

    return "".join(result)


# ============================================================
# 11. TOKENIZER TEST
# ============================================================

print("=" * 60)
print("TOKENIZER TEST")
print("=" * 60)
print()

test_text = (
    "In the beginning God created\n"
    "And God said, Let there be light.\n"
    "The Lord is my shepherd."
)

print("Original:")
print(repr(test_text))
print()

encoded = encode_text(
    test_text
)

print(
    "Token count:",
    len(encoded)
)

print()

decoded = decode(
    encoded
)

print("Decoded:")
print(repr(decoded))
print()

if decoded == test_text:

    print("[OK] Round-trip test PASSED")

else:

    print("[ERROR] Round-trip test FAILED")

    print()
    print("Original:")
    print(repr(test_text))

    print()
    print("Decoded:")
    print(repr(decoded))

    raise RuntimeError(
        "Tokenizer round-trip test failed."
    )

print()


# ============================================================
# 12. TEST WORD TOKENIZATION
# ============================================================

print("=" * 60)
print("WORD TOKENIZATION TEST")
print("=" * 60)
print()

test_words = [
    "beginning",
    "created",
    "righteousness",
    "Jerusalem",
    "commandment",
    "Jesus",
    "Lord"
]

for word in test_words:

    ids = encode_text(word)

    print(
        word,
        "->",
        len(ids),
        "tokens"
    )

print()


# ============================================================
# 13. SAVE TOKENIZER
# ============================================================

tokenizer_data = {

    "type": "BPE",

    "version": "4.0",

    "vocab_size": len(tokens),

    "tokens": tokens,

    "merges": merge_rules

}


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        tokenizer_data,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("=" * 60)
print("V4 TOKENIZER SAVED")
print("=" * 60)
print()

print(
    "Vocabulary:",
    len(tokens)
)

print(
    "Merges:",
    len(merge_rules)
)

print(
    "Output:",
    OUTPUT_FILE
)

print()

print("Next step:")
print("Create dataset_v4.py")

print()
print("=" * 60)
print("SCRIPTURELM V4 TOKENIZER COMPLETE")
print("=" * 60)