import json
import re
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

# build_documents.py is inside:
# ScriptureLM-SelfBuilt/src/rag/
#
# parent        -> src
# parent.parent -> ScriptureLM-SelfBuilt
#
BASE_DIR = Path(__file__).resolve().parent.parent.parent

CORPUS_FILE = BASE_DIR / "data" / "raw" / "bible_corpus.txt"
OUTPUT_DIR = BASE_DIR / "data" / "rag"
OUTPUT_FILE = OUTPUT_DIR / "documents.json"


# --------------------------------------------------
# VERSE PATTERN
# --------------------------------------------------

# Example:
# [1 Chronicles 1:1] Adam, Sheth, Enosh,

VERSE_PATTERN = re.compile(
    r"^\[([^\]]+)\]\s*(.*)$"
)


# --------------------------------------------------
# BUILD DOCUMENTS
# --------------------------------------------------

def build_documents():

    print("=" * 60)
    print("SCRIPTURELM RAG DOCUMENT BUILDER")
    print("=" * 60)

    # Check whether the Bible corpus exists
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Bible corpus not found: {CORPUS_FILE}"
        )

    # Create RAG output directory if it doesn't exist
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    # --------------------------------------------------
    # READ BIBLE CORPUS
    # --------------------------------------------------

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Match:
            # [Book Chapter:Verse] Text
            match = VERSE_PATTERN.match(line)

            if not match:

                print(
                    f"Warning: Could not parse "
                    f"line {line_number}"
                )

                continue

            # Extract reference and verse text
            reference = match.group(1)
            text = match.group(2).strip()

            # --------------------------------------------------
            # SPLIT BOOK AND CHAPTER:VERSE
            # --------------------------------------------------

            parts = reference.rsplit(
                " ",
                1
            )

            if len(parts) != 2:

                print(
                    f"Warning: Invalid reference "
                    f"at line {line_number}: "
                    f"{reference}"
                )

                continue

            book = parts[0]
            chapter_verse = parts[1]

            # Split chapter and verse
            try:

                chapter, verse = chapter_verse.split(
                    ":"
                )

            except ValueError:

                print(
                    f"Warning: Invalid chapter/verse "
                    f"at line {line_number}: "
                    f"{reference}"
                )

                continue

            # --------------------------------------------------
            # CREATE DOCUMENT
            # --------------------------------------------------

            document = {
                "id": reference,
                "book": book,
                "chapter": int(chapter),
                "verse": int(verse),
                "reference": reference,
                "text": text
            }

            documents.append(document)

    # --------------------------------------------------
    # SAVE DOCUMENTS
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print(
        f"Total verses processed: "
        f"{len(documents)}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # SHOW FIRST DOCUMENT
    # --------------------------------------------------

    if documents:

        print()
        print("First document:")

        print(
            json.dumps(
                documents[0],
                ensure_ascii=False,
                indent=2
            )
        )

        # --------------------------------------------------
        # SHOW LAST DOCUMENT
        # --------------------------------------------------

        print()
        print("Last document:")

        print(
            json.dumps(
                documents[-1],
                ensure_ascii=False,
                indent=2
            )
        )

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("DOCUMENT BUILD COMPLETE")
    print("=" * 60)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    build_documents()