import json
import re
import torch
from torch.utils.data import Dataset


# ============================================================
# SCRIPTURELM V4
# DATASET
# ============================================================

TOKENIZER_FILE = "../data/processed/tokenizer_v4.json"
BIBLE_FILE = "../data/raw/bible_corpus.txt"

DATASET_OUTPUT = "../data/processed/dataset_v4.pt"

BLOCK_SIZE = 256


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("ScriptureLM V4 - Dataset")
print("=" * 60)
print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading V4 tokenizer...")

with open(
    TOKENIZER_FILE,
    "r",
    encoding="utf-8"
) as file:

    tokenizer_data = json.load(file)


vocabulary = tokenizer_data["tokens"]
merges = tokenizer_data["merges"]

vocab_size = len(vocabulary)


print("Vocabulary:", vocab_size)
print("BPE merges:", len(merges))
print()


# ============================================================
# TOKEN MAPPINGS
# ============================================================

token_to_id = {
    token: index
    for index, token
    in enumerate(vocabulary)
}


id_to_token = {
    index: token
    for token, index
    in token_to_id.items()
}


# ============================================================
# BPE MERGE
# ============================================================

def merge_pair(
    sequence,
    pair
):

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
# ENCODE
# ============================================================

def encode_text(text):

    sequence = list(text)

    for pair in merges:

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
# DECODE
# ============================================================

def decode_tokens(token_ids):

    result = []

    for token_id in token_ids:

        if int(token_id) not in id_to_token:

            raise ValueError(
                "Unknown token ID: "
                + str(token_id)
            )

        result.append(
            id_to_token[int(token_id)]
        )

    return "".join(result)


# ============================================================
# LOAD BIBLE CORPUS
# ============================================================

print("Loading Bible corpus...")

with open(
    BIBLE_FILE,
    "r",
    encoding="utf-8"
) as file:

    original_text = file.read()


print(
    "Original characters:",
    len(original_text)
)

print()


# ============================================================
# CLEAN VERSE REFERENCES
# ============================================================

print("Cleaning verse references...")


clean_text = re.sub(
    r"\[[^\]\n]+\]",
    "",
    original_text
)


# Normalize spaces and tabs.

clean_text = re.sub(
    r"[ \t]+",
    " ",
    clean_text
)


# Prevent excessive blank lines.

clean_text = re.sub(
    r"\n{3,}",
    "\n\n",
    clean_text
)


print(
    "Cleaned characters:",
    len(clean_text)
)

print()


# ============================================================
# SPLIT INTO LINES
# ============================================================

lines = clean_text.splitlines(
    keepends=True
)


print(
    "Total lines:",
    len(lines)
)

print()


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_position = int(
    0.90 * len(lines)
)


training_lines = lines[
    :split_position
]


validation_lines = lines[
    split_position:
]


print(
    "Training lines:",
    len(training_lines)
)

print(
    "Validation lines:",
    len(validation_lines)
)

print()


# ============================================================
# CREATE TEXT
# ============================================================

training_text = "".join(
    training_lines
)

validation_text = "".join(
    validation_lines
)


# ============================================================
# ENCODE TRAINING DATA
# ============================================================

print("Encoding training data...")

training_tokens = encode_text(
    training_text
)


print(
    "Training tokens:",
    len(training_tokens)
)

print()


# ============================================================
# ENCODE VALIDATION DATA
# ============================================================

print("Encoding validation data...")

validation_tokens = encode_text(
    validation_text
)


print(
    "Validation tokens:",
    len(validation_tokens)
)

print()


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

train_data = torch.tensor(
    training_tokens,
    dtype=torch.long
)


validation_data = torch.tensor(
    validation_tokens,
    dtype=torch.long
)


# ============================================================
# DATASET CLASS
# ============================================================

class BibleDataset(Dataset):

    def __init__(
        self,
        data,
        block_size
    ):

        self.data = data

        self.block_size = block_size


    def __len__(self):

        return (
            len(self.data)
            -
            self.block_size
        )


    def __getitem__(
        self,
        index
    ):

        x = self.data[
            index:
            index + self.block_size
        ]


        y = self.data[
            index + 1:
            index + self.block_size + 1
        ]


        return x, y


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = BibleDataset(
    train_data,
    BLOCK_SIZE
)


validation_dataset = BibleDataset(
    validation_data,
    BLOCK_SIZE
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("=" * 60)
print("V4 DATASET INFORMATION")
print("=" * 60)
print()


print(
    "Vocabulary:",
    vocab_size
)


print(
    "Training tokens:",
    len(train_data)
)


print(
    "Validation tokens:",
    len(validation_data)
)


print(
    "Training samples:",
    len(train_dataset)
)


print(
    "Validation samples:",
    len(validation_dataset)
)


print(
    "Block size:",
    BLOCK_SIZE
)

print()


# ============================================================
# TRAINING SAMPLE TEST
# ============================================================

print("=" * 60)
print("TRAINING SAMPLE TEST")
print("=" * 60)
print()


x, y = train_dataset[0]


print(
    "Input shape:",
    x.shape
)


print(
    "Target shape:",
    y.shape
)


# ============================================================
# SHIFT TEST
# ============================================================

shift_test = torch.equal(
    x[1:],
    y[:-1]
)


print(
    "Shift test:",
    shift_test
)

print()


# ============================================================
# TOKEN TEST
# ============================================================

print("First 30 input tokens:")

print(
    x[:30]
)

print()


print("First 30 target tokens:")

print(
    y[:30]
)

print()


# ============================================================
# DECODE SAMPLE
# ============================================================

decoded_sample = decode_tokens(
    x[:100]
)


print("=" * 60)
print("DECODED SAMPLE")
print("=" * 60)
print()


print(
    decoded_sample
)

print()


# ============================================================
# TOKENIZER ROUND-TRIP TEST
# ============================================================

test_text = (
    "In the beginning God created "
    "the heaven and the earth.\n"
    "And God said, Let there be light."
)


test_ids = encode_text(
    test_text
)


test_decoded = decode_tokens(
    test_ids
)


print("=" * 60)
print("TOKENIZER ROUND-TRIP TEST")
print("=" * 60)
print()


print(
    "Original:",
    repr(test_text)
)

print()


print(
    "Decoded:",
    repr(test_decoded)
)

print()


print(
    "Match:",
    test_text == test_decoded
)

print()


# ============================================================
# FINAL CHECKS
# ============================================================

if not shift_test:

    raise RuntimeError(
        "Dataset shift test FAILED."
    )


if test_text != test_decoded:

    raise RuntimeError(
        "Tokenizer round-trip test FAILED."
    )


if vocab_size != 2048:

    raise RuntimeError(
        "Unexpected V4 vocabulary size: "
        + str(vocab_size)
    )


print("=" * 60)
print("V4 DATASET TESTS PASSED")
print("=" * 60)
print()


print("Vocabulary:", vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(validation_data))
print("Block size:", BLOCK_SIZE)

print()


# ============================================================
# SAVE V4 DATASET
# ============================================================

print("=" * 60)
print("SAVING V4 DATASET")
print("=" * 60)
print()


dataset_data = {

    "train": train_data,

    "val": validation_data,

    "vocab_size": vocab_size,

    "block_size": BLOCK_SIZE

}


torch.save(
    dataset_data,
    DATASET_OUTPUT
)


print(
    "V4 dataset saved:"
)

print(
    DATASET_OUTPUT
)

print()


print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(validation_data)
)

print()


# ============================================================
# VERIFY SAVED DATASET
# ============================================================

if not torch.cuda.is_available():

    print(
        "Training device will be CPU."
    )


if not torch.tensor(
    dataset_data["train"]
).equal(train_data):

    raise RuntimeError(
        "Saved training dataset verification FAILED."
    )


if not torch.tensor(
    dataset_data["val"]
).equal(validation_data):

    raise RuntimeError(
        "Saved validation dataset verification FAILED."
    )


print(
    "[OK] Saved dataset verification PASSED"
)

print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("SCRIPTURELM V4 DATASET COMPLETE")
print("=" * 60)

print()

print(
    "Dataset file:"
)

print(
    DATASET_OUTPUT
)

print()

print(
    "Next step:"
)

print(
    "Train V4 model"
)

print()

print("=" * 60)
print("READY FOR V4 TRAINING")
print("=" * 60)