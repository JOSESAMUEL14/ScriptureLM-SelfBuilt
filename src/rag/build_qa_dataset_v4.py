import json
import random
import re
from collections import Counter
from pathlib import Path


# ============================================================
# SCRIPTURELM - QA DATASET V4
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CORPUS_FILE = BASE_DIR / "data" / "raw" / "bible_corpus.txt"
OUTPUT_DIR = BASE_DIR / "data" / "rag"
OUTPUT_FILE = OUTPUT_DIR / "qa_dataset_v4.json"

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

TARGET_TOTAL = 12000


# ============================================================
# PEOPLE
# ============================================================

PEOPLE = {
    "Moses": ["moses"],
    "David": ["david"],
    "Abraham": ["abraham", "abram"],
    "Isaac": ["isaac"],
    "Jacob": ["jacob"],
    "Joseph": ["joseph"],
    "Noah": ["noah"],
    "Solomon": ["solomon"],
    "Joshua": ["joshua"],
    "Samuel": ["samuel"],
    "Daniel": ["daniel"],
    "Elijah": ["elijah"],
    "Elisha": ["elisha"],
    "Peter": ["peter"],
    "Paul": ["paul"],
    "John": ["john"],
    "Jesus": ["jesus", "christ"],
    "Mary": ["mary"],
    "Martha": ["martha"],
    "Lazarus": ["lazarus"],
    "Esther": ["esther"],
    "Ruth": ["ruth"],
    "Job": ["job"],
    "Aaron": ["aaron"],
}


# ============================================================
# TOPICS
# ============================================================

TOPICS = {
    "love": ["love", "loveth", "loved"],
    "faith": ["faith", "believe", "believed", "believeth"],
    "forgiveness": ["forgive", "forgiven", "forgiveness"],
    "prayer": ["pray", "prayer", "prayed"],
    "wisdom": ["wisdom", "wise"],
    "peace": ["peace"],
    "sin": ["sin", "sins", "sinned"],
    "salvation": ["salvation", "save", "saved"],
    "hope": ["hope"],
    "mercy": ["mercy", "merciful"],
    "obedience": ["obey", "obeyed", "obedience"],
    "righteousness": ["righteousness", "righteous"],
    "repentance": ["repent", "repentance"],
    "eternal life": ["eternal life"],
    "creation": ["created", "creation", "create"],
    "resurrection": ["resurrection", "risen", "raised"],
    "heaven": ["heaven", "heavens"],
    "marriage": ["marriage", "husband", "wife"],
    "family": ["family", "father", "mother", "children"],
    "humility": ["humble", "humility"],
    "justice": ["justice"],
    "grace": ["grace"],
    "truth": ["truth"],
    "joy": ["joy", "rejoice"],
    "patience": ["patience", "patient"],
    "kindness": ["kindness", "kind"],
    "courage": ["courage", "strong", "strength"],
    "fear": ["fear", "afraid"],
    "anger": ["anger", "angry", "wrath"],
    "temptation": ["tempt", "temptation"],
    "worship": ["worship"],
    "praise": ["praise", "praised"],
    "thanksgiving": ["thanksgiving", "thanks"],
    "fasting": ["fast", "fasting"],
    "healing": ["heal", "healed", "healing"],
    "miracles": ["miracle", "miracles", "wonder"],
    "suffering": ["suffer", "suffering", "affliction"],
    "persecution": ["persecuted", "persecution"],
    "discipleship": ["disciple", "disciples"],
    "church": ["church"],
    "kingdom of God": ["kingdom of god"],
    "gospel": ["gospel"],
    "Holy Spirit": ["holy spirit"],
    "commandments": ["commandment", "commandments"],
    "judgment": ["judgment", "judge"],
    "money": ["money"],
    "riches": ["riches", "rich"],
    "poverty": ["poor", "poverty"],
    "work": ["work", "labour"],
}


# ============================================================
# NATURAL QUESTION TEMPLATES
# ============================================================

PERSON_QUESTIONS = [
    "Who was {person}?",
    "Who is {person} according to the Bible?",
    "What does the Bible say about {person}?",
    "What do we know about {person} from Scripture?",
    "What is {person} known for in the Bible?",
    "How is {person} described in Scripture?",
    "What happened in the life of {person}?",
    "What role did {person} have in the Bible?",
]


TOPIC_QUESTIONS = [
    "What does the Bible say about {topic}?",
    "What does Scripture teach about {topic}?",
    "How does the Bible address {topic}?",
    "What guidance does the Bible give about {topic}?",
    "What can we learn from Scripture about {topic}?",
    "How is {topic} described in the Bible?",
    "What does Scripture reveal about {topic}?",
    "What Bible passages discuss {topic}?",
]


# ============================================================
# QUESTION TYPES
# ============================================================

QUESTION_TYPES = [
    "person",
    "topic",
]


# ============================================================
# LOAD CORPUS
# ============================================================

def load_verses():

    verses = []

    pattern = re.compile(
        r"^\[([^\]]+)\]\s*(.*)$"
    )

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            match = pattern.match(line)

            if not match:
                continue

            reference = match.group(1).strip()

            text = match.group(2).strip()

            if not text:
                continue

            # Encoding cleanup.
            replacements = {
                "ΓÇÖ": "'",
                "ΓÇô": "-",
                "ΓÇ£": '"',
                "ΓÇ¥": '"',
                "â€™": "'",
                "â€“": "-",
                "â€œ": '"',
                "â€": '"',
            }

            for old, new in replacements.items():
                text = text.replace(old, new)

            verses.append({
                "reference": reference,
                "text": text,
            })

    return verses


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    text = text.lower()

    text = text.replace("’", "'")

    text = re.sub(
        r"[^a-z0-9\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# KEYWORD MATCHING
# ============================================================

def keyword_score(text, keywords):

    normalized = normalize(text)

    score = 0

    words = set(
        normalized.split()
    )

    for keyword in keywords:

        keyword = keyword.lower()

        if " " in keyword:

            if keyword in normalized:
                score += 4

        else:

            if keyword in words:
                score += 2

    return score


# ============================================================
# FIND RELEVANT VERSES
# ============================================================

def find_relevant(
    verses,
    keywords,
    limit=30
):

    results = []

    for index, verse in enumerate(verses):

        score = keyword_score(
            verse["text"],
            keywords
        )

        if score > 0:

            results.append({
                "index": index,
                "score": score,
                "verse": verse,
            })

    results.sort(
        key=lambda x: (
            -x["score"],
            x["index"]
        )
    )

    return results[:limit]


# ============================================================
# PARSE REFERENCE
# ============================================================

def parse_reference(reference):

    match = re.match(
        r"^(.*)\s+(\d+):(\d+)$",
        reference
    )

    if not match:
        return None, None, None

    book = match.group(1)

    chapter = int(
        match.group(2)
    )

    verse = int(
        match.group(3)
    )

    return book, chapter, verse


# ============================================================
# BUILD MULTI-VERSE CONTEXT
# ============================================================

def build_context(
    verses,
    target_index,
    related_indices=None
):

    candidates = []

    target = verses[target_index]

    target_book, target_chapter, _ = parse_reference(
        target["reference"]
    )

    # Previous / next verses from same chapter.
    for offset in [-2, -1, 1, 2]:

        index = target_index + offset

        if index < 0 or index >= len(verses):
            continue

        verse = verses[index]

        book, chapter, _ = parse_reference(
            verse["reference"]
        )

        if (
            book == target_book
            and chapter == target_chapter
        ):
            candidates.append(index)

    # Add related verses.
    if related_indices:

        for index in related_indices:

            if index == target_index:
                continue

            if index not in candidates:
                candidates.append(index)

    # Target must always be included.
    selected_indices = [
        target_index
    ]

    for index in candidates:

        if index in selected_indices:
            continue

        selected_indices.append(index)

        if len(selected_indices) >= 5:
            break

    # Shuffle context order.
    random.shuffle(
        selected_indices
    )

    lines = []

    for index in selected_indices:

        verse = verses[index]

        lines.append(
            f"{verse['reference']}: "
            f"{verse['text']}"
        )

    return "\n".join(lines)


# ============================================================
# BUILD ANSWER
# ============================================================

def build_answer(
    target_verse,
    keywords
):

    text = target_verse["text"].strip()

    if not text:
        return ""

    # Find sentences containing relevant keywords.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    relevant = []

    for sentence in sentences:

        if keyword_score(
            sentence,
            keywords
        ) > 0:

            relevant.append(
                sentence.strip()
            )

    # If relevant sentences exist,
    # use them as the answer.
    if relevant:

        return " ".join(
            relevant[:3]
        )

    return text


# ============================================================
# CREATE PERSON DATA
# ============================================================

def create_person_examples(
    verses
):

    examples = []

    print()
    print("Creating person QA...")

    for person, keywords in PEOPLE.items():

        print(
            f"  {person}"
        )

        matches = find_relevant(
            verses,
            keywords,
            limit=40
        )

        if not matches:
            continue

        for match in matches:

            target_index = match["index"]

            related_indices = [
                item["index"]
                for item in matches
                if item["index"] != target_index
            ]

            context = build_context(
                verses,
                target_index,
                related_indices
            )

            answer = build_answer(
                verses[target_index],
                keywords
            )

            if not answer:
                continue

            for template in PERSON_QUESTIONS:

                question = template.format(
                    person=person
                )

                examples.append({

                    "question": question,

                    "context": context,

                    "answer": answer,

                    "type": "person",

                    "entity": person,

                    "source_reference":
                        verses[target_index]["reference"]

                })

    return examples


# ============================================================
# CREATE TOPIC DATA
# ============================================================

def create_topic_examples(
    verses
):

    examples = []

    print()
    print("Creating topic QA...")

    for topic, keywords in TOPICS.items():

        print(
            f"  {topic}"
        )

        matches = find_relevant(
            verses,
            keywords,
            limit=40
        )

        if not matches:
            continue

        for match in matches:

            target_index = match["index"]

            related_indices = [
                item["index"]
                for item in matches
                if item["index"] != target_index
            ]

            context = build_context(
                verses,
                target_index,
                related_indices
            )

            answer = build_answer(
                verses[target_index],
                keywords
            )

            if not answer:
                continue

            for template in TOPIC_QUESTIONS:

                question = template.format(
                    topic=topic
                )

                examples.append({

                    "question": question,

                    "context": context,

                    "answer": answer,

                    "type": "topic",

                    "entity": topic,

                    "source_reference":
                        verses[target_index]["reference"]

                })

    return examples


# ============================================================
# DEDUPLICATION
# ============================================================

def remove_duplicates(
    examples
):

    seen = set()

    unique = []

    for example in examples:

        key = (
            example["question"].strip().lower(),
            example["context"].strip().lower(),
            example["answer"].strip().lower(),
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            example
        )

    return unique


# ============================================================
# CREATE EXTRA NATURAL VARIATIONS
# ============================================================

def create_question_variations(
    examples
):

    variations = []

    for example in examples:

        question = example["question"]

        replacements = [

            (
                "What does the Bible say about",
                "What does the Bible teach about"
            ),

            (
                "What does Scripture teach about",
                "What does Scripture reveal about"
            ),

            (
                "How does the Bible address",
                "How does Scripture address"
            ),

            (
                "What can we learn from Scripture about",
                "What can we learn from the Bible about"
            ),

        ]

        for old, new in replacements:

            if old in question:

                new_question = question.replace(
                    old,
                    new
                )

                new_example = example.copy()

                new_example[
                    "question"
                ] = new_question

                variations.append(
                    new_example
                )

    return variations


# ============================================================
# SPLIT BY SOURCE REFERENCE
# ============================================================

def split_by_reference(
    examples
):

    groups = {}

    for example in examples:

        reference = example[
            "source_reference"
        ]

        if reference not in groups:

            groups[reference] = []

        groups[
            reference
        ].append(example)

    references = list(
        groups.keys()
    )

    random.shuffle(
        references
    )

    total = len(
        references
    )

    train_end = int(
        total * 0.80
    )

    validation_end = int(
        total * 0.90
    )

    train_refs = references[
        :train_end
    ]

    validation_refs = references[
        train_end:validation_end
    ]

    test_refs = references[
        validation_end:
    ]

    train = []

    validation = []

    test = []

    for reference in train_refs:

        train.extend(
            groups[reference]
        )

    for reference in validation_refs:

        validation.extend(
            groups[reference]
        )

    for reference in test_refs:

        test.extend(
            groups[reference]
        )

    return (
        train,
        validation,
        test
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    examples,
    train,
    validation,
    test
):

    print()
    print("=" * 70)
    print("V4 DATASET STATISTICS")
    print("=" * 70)

    print()

    print(
        "Total:",
        len(examples)
    )

    print(
        "Train:",
        len(train)
    )

    print(
        "Validation:",
        len(validation)
    )

    print(
        "Test:",
        len(test)
    )

    print()

    counts = Counter(
        example["type"]
        for example in examples
    )

    print(
        "Types:"
    )

    for key, value in counts.items():

        print(
            f"  {key}: {value}"
        )

    print()

    if examples:

        lengths = [
            len(x["answer"])
            for x in examples
        ]

        print(
            "Answer length:"
        )

        print(
            "  Min:",
            min(lengths)
        )

        print(
            "  Max:",
            max(lengths)
        )

        print(
            "  Average:",
            round(
                sum(lengths)
                / len(lengths),
                2
            )
        )

    print()

    references = set(
        x["source_reference"]
        for x in examples
    )

    print(
        "Unique source verses:",
        len(references)
    )


# ============================================================
# SHOW SAMPLES
# ============================================================

def show_samples(
    examples,
    count=5
):

    print()
    print("=" * 70)
    print("SAMPLE V4 QUESTIONS")
    print("=" * 70)

    samples = random.sample(
        examples,
        min(
            count,
            len(examples)
        )
    )

    for number, example in enumerate(
        samples,
        1
    ):

        print()
        print(
            f"--- SAMPLE {number} ---"
        )

        print(
            "Type:",
            example["type"]
        )

        print(
            "Question:",
            example["question"]
        )

        print(
            "Context:"
        )

        print(
            example["context"]
        )

        print(
            "Answer:"
        )

        print(
            example["answer"]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SCRIPTURELM V4 QA DATASET BUILDER")
    print("=" * 70)

    print()

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"Corpus not found:\n{CORPUS_FILE}"
        )

    print(
        "Loading Bible corpus..."
    )

    verses = load_verses()

    print(
        "Loaded verses:",
        len(verses)
    )

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    person_examples = create_person_examples(
        verses
    )

    print(
        "Person examples:",
        len(person_examples)
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    topic_examples = create_topic_examples(
        verses
    )

    print(
        "Topic examples:",
        len(topic_examples)
    )

    # --------------------------------------------------------
    # NATURAL VARIATIONS
    # --------------------------------------------------------

    base_examples = (
        person_examples
        +
        topic_examples
    )

    variations = create_question_variations(
        base_examples
    )

    print(
        "Additional natural variations:",
        len(variations)
    )

    examples = (
        base_examples
        +
        variations
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    examples = remove_duplicates(
        examples
    )

    print(
        "After duplicate removal:",
        len(examples)
    )

    # --------------------------------------------------------
    # IF DATASET IS TOO SMALL
    # --------------------------------------------------------

    if len(examples) < TARGET_TOTAL:

        print()
        print(
            "Dataset is below target."
        )

        print(
            "Adding additional grounded examples..."
        )

        # Use all verses as fallback,
        # but still without another AI.
        for verse in verses:

            if len(examples) >= TARGET_TOTAL:
                break

            question_templates = [

                "What is recorded in {reference}?",
                "What happened according to {reference}?",
                "What does {reference} tell us?",
                "What is described in {reference}?",

            ]

            for template in question_templates:

                if len(examples) >= TARGET_TOTAL:
                    break

                question = template.format(
                    reference=verse["reference"]
                )

                examples.append({

                    "question": question,

                    "context": (
                        f"{verse['reference']}: "
                        f"{verse['text']}"
                    ),

                    "answer": verse["text"],

                    "type": "grounded_verse",

                    "entity": verse["reference"],

                    "source_reference":
                        verse["reference"]

                })

    # --------------------------------------------------------
    # DEDUPE AGAIN
    # --------------------------------------------------------

    examples = remove_duplicates(
        examples
    )

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(
        examples
    )

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    if len(examples) > TARGET_TOTAL:

        examples = examples[
            :TARGET_TOTAL
        ]

    print()
    print(
        "Final examples:",
        len(examples)
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train, validation, test = split_by_reference(
        examples
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = {

        "train": train,

        "validation": validation,

        "test": test,

        "metadata": {

            "version": "v4",

            "total_examples":
                len(examples),

            "source":
                "Bible corpus",

            "generation":
                "deterministic Python rules",

            "external_ai":
                False,

            "external_api":
                False,

            "random_seed":
                RANDOM_SEED

        }

    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2
        )

    print_statistics(
        examples,
        train,
        validation,
        test
    )

    show_samples(
        examples,
        count=5
    )

    print()
    print("=" * 70)
    print("SCRIPTURELM V4 QA DATASET BUILD COMPLETE")
    print("=" * 70)

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()