import json
import torch
import torch.nn.functional as F

from model_v4 import LanguageModel


# ============================================================
# SCRIPTURELM V4 - TEXT GENERATION
# ============================================================

MODEL_FILE = "../data/processed/scripturelm_v4_best.pth"
TOKENIZER_FILE = "../data/processed/tokenizer_v4.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK_SIZE = 256


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 60)
print("SCRIPTURELM V4 GENERATION")
print("=" * 60)
print()

print("Device:", DEVICE)

print()
print("Loading tokenizer...")

with open(
    TOKENIZER_FILE,
    "r",
    encoding="utf-8"
) as file:

    tokenizer_data = json.load(file)


tokens = tokenizer_data["tokens"]
merges = tokenizer_data["merges"]

VOCAB_SIZE = tokenizer_data["vocab_size"]

token_to_id = {
    token: index
    for index, token in enumerate(tokens)
}

id_to_token = {
    index: token
    for index, token in enumerate(tokens)
}


print("Vocabulary size:", VOCAB_SIZE)
print("BPE merges:", len(merges))


# ============================================================
# TOKENIZER FUNCTIONS
# ============================================================

def merge_pair(sequence, pair):

    new_sequence = []

    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):

            new_sequence.append(
                pair[0] + pair[1]
            )

            i += 2

        else:

            new_sequence.append(
                sequence[i]
            )

            i += 1

    return new_sequence


def encode(text):

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
                f"Unknown token during encoding: {repr(token)}"
            )

        token_ids.append(
            token_to_id[token]
        )

    return token_ids


def decode(token_ids):

    result = []

    for token_id in token_ids:

        result.append(
            id_to_token[int(token_id)]
        )

    return "".join(result)


# ============================================================
# TOKENIZER TEST
# ============================================================

print()
print("Testing tokenizer...")

test_text = (
    "In the beginning God created "
    "the heaven and the earth."
)

test_ids = encode(test_text)

test_decoded = decode(test_ids)

print(
    "Tokenizer test:",
    test_decoded == test_text
)

if test_decoded != test_text:

    raise RuntimeError(
        "Tokenizer round-trip test FAILED."
    )

print("Tokenizer test PASSED.")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading V4 model...")

model = LanguageModel()

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)

model.eval()

print("V4 model loaded successfully.")


# ============================================================
# GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.9
):

    token_ids = encode(prompt)

    if len(token_ids) == 0:

        raise ValueError(
            "Prompt produced no tokens."
        )


    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


    for _ in range(max_new_tokens):

        # Keep only the most recent context
        idx_cond = idx[:, -BLOCK_SIZE:]


        # Model prediction
        logits, _ = model(
            idx_cond
        )


        # Last token logits
        logits = logits[:, -1, :]


        # Temperature
        logits = logits / temperature


        # Top-K filtering
        if top_k is not None:

            k = min(
                top_k,
                logits.size(-1)
            )

            values, _ = torch.topk(
                logits,
                k
            )

            minimum_value = values[
                :, -1
            ].unsqueeze(-1)

            logits = torch.where(
                logits < minimum_value,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits
            )


        # Top-P filtering
        if top_p is not None:

            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True
            )

            sorted_probabilities = F.softmax(
                sorted_logits,
                dim=-1
            )

            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim=-1
            )

            remove_mask = (
                cumulative_probabilities > top_p
            )

            # Keep at least one token
            remove_mask[:, 0] = False

            sorted_logits = sorted_logits.masked_fill(
                remove_mask,
                float("-inf")
            )

            logits = torch.full_like(
                logits,
                float("-inf")
            )

            logits.scatter_(
                -1,
                sorted_indices,
                sorted_logits
            )


        # Convert to probabilities
        probabilities = F.softmax(
            logits,
            dim=-1
        )


        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )


        # Append token
        idx = torch.cat(
            (
                idx,
                next_token
            ),
            dim=1
        )


    return decode(
        idx[0].tolist()
    )


# ============================================================
# GENERATION TESTS
# ============================================================

print()
print("=" * 60)
print("SCRIPTURELM V4 GENERATION TEST")
print("=" * 60)

print()

print("Temperature: 0.8")
print("Top-K: 40")
print("Top-P: 0.9")

print()


prompts = [

    "In the beginning",

    "The Lord is",

    "And God said",

    "Jesus said",

    "For God so loved"

]


for prompt in prompts:

    print("-" * 60)

    print("Prompt:")

    print(prompt)

    print()

    try:

        generated = generate(
            prompt,
            max_new_tokens=150,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )

        print("Generated:")

        print(generated)

    except Exception as error:

        print(
            "Generation error:",
            error
        )

    print()


print("=" * 60)

print("V4 GENERATION TEST COMPLETE")

print("=" * 60)