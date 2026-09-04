import json
import math
import os
import random

import torch

from model_v4 import (
    LanguageModel,
    VOCAB_SIZE,
    BLOCK_SIZE
)


# ============================================================
# SCRIPTURELM V4 - TRAINING
# CPU FRIENDLY + CHECKPOINT / RESUME
# ============================================================

TOKENIZER_PATH = "../data/processed/tokenizer_v4.json"

BEST_CHECKPOINT = "../data/processed/scripturelm_v4_best.pth"
LAST_CHECKPOINT = "../data/processed/scripturelm_v4_checkpoint.pth"
FINAL_CHECKPOINT = "../data/processed/scripturelm_v4_final.pth"


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 16

EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE *
    GRADIENT_ACCUMULATION_STEPS
)

TOTAL_STEPS = 5000

VALIDATION_INTERVAL = 100

PRINT_INTERVAL = 10

LEARNING_RATE = 3e-4

MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1

GRADIENT_CLIP = 1.0

WARMUP_STEPS = 300


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = "cuda"

else:

    device = "cpu"


print(
    "Training device:",
    device
)

print()


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = "../data/processed/dataset_v4.pt"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("SCRIPTURELM V4 TRAINING")
print("=" * 60)

print()

print("Loading V4 dataset...")

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        "V4 dataset not found:\n"
        + DATASET_PATH
        + "\n\n"
        "Run dataset_v4.py first."
    )


dataset = torch.load(
    DATASET_PATH,
    map_location="cpu"
)


# ============================================================
# SUPPORT DIFFERENT DATASET KEY NAMES
# ============================================================

if isinstance(dataset, dict):

    if "train" in dataset:

        train_data = dataset["train"]

    elif "train_tokens" in dataset:

        train_data = dataset["train_tokens"]

    else:

        raise KeyError(
            "Could not find training data in dataset."
        )


    if "val" in dataset:

        val_data = dataset["val"]

    elif "validation" in dataset:

        val_data = dataset["validation"]

    elif "val_tokens" in dataset:

        val_data = dataset["val_tokens"]

    elif "validation_tokens" in dataset:

        val_data = dataset["validation_tokens"]

    else:

        raise KeyError(
            "Could not find validation data in dataset."
        )

else:

    raise TypeError(
        "Unexpected V4 dataset format."
    )


train_data = train_data.long()

val_data = val_data.long()


print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(val_data)
)

print()


# ============================================================
# DATASET FUNCTION
# ============================================================

def get_batch(data):

    max_start = len(data) - BLOCK_SIZE - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (
            BATCH_SIZE,
        )
    )

    x = torch.stack(
        [
            data[
                start:
                start + BLOCK_SIZE
            ]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            data[
                start + 1:
                start + BLOCK_SIZE + 1
            ]
            for start in starts
        ]
    )

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def estimate_validation_loss(
    model,
    batches=20
):

    model.eval()

    losses = []

    for _ in range(batches):

        x, y = get_batch(
            val_data
        )

        _, loss = model(
            x,
            y
        )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(step):

    # Warmup
    if step < WARMUP_STEPS:

        return LEARNING_RATE * (
            step + 1
        ) / WARMUP_STEPS


    # Progress after warmup
    progress = (
        step - WARMUP_STEPS
    ) / max(
        1,
        TOTAL_STEPS - WARMUP_STEPS
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress
        )
    )


    # Cosine decay
    cosine = 0.5 * (
        1.0
        +
        math.cos(
            math.pi * progress
        )
    )


    return (
        MIN_LEARNING_RATE
        +
        (
            LEARNING_RATE
            -
            MIN_LEARNING_RATE
        )
        *
        cosine
    )


# ============================================================
# MODEL
# ============================================================

print("Creating V4 model...")

model = LanguageModel()

model = model.to(device)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    "V4 model parameters:",
    parameter_count
)

print()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# RESUME STATE
# ============================================================

start_step = 0

best_val_loss = float("inf")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if os.path.exists(LAST_CHECKPOINT):

    print(
        "Existing V4 checkpoint found."
    )

    print(
        LAST_CHECKPOINT
    )

    print()

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location=device
    )


    # --------------------------------------------------------
    # New checkpoint format
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if "optimizer_state_dict" in checkpoint:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )


        start_step = checkpoint.get(
            "step",
            0
        )

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )


        print(
            "Checkpoint loaded successfully."
        )

        print(
            "Starting from step:",
            start_step
        )

        print(
            "Best validation loss:",
            best_val_loss
        )


    # --------------------------------------------------------
    # Model-only checkpoint fallback
    # --------------------------------------------------------

    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "Model checkpoint loaded."
        )

        print(
            "Optimizer state unavailable."
        )


    print()


else:

    print(
        "No previous checkpoint found."
    )

    print(
        "Starting V4 training from step 0."
    )

    print()


# ============================================================
# SAFETY CHECK
# ============================================================

if start_step >= TOTAL_STEPS:

    print(
        "Training target already reached."
    )

    print(
        "Current step:",
        start_step
    )

    print(
        "Target step:",
        TOTAL_STEPS
    )

    print()

    print(
        "No additional training required."
    )

    raise SystemExit


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

print()

print(
    "Starting step:",
    start_step
)

print(
    "Target step:",
    TOTAL_STEPS
)

print(
    "Remaining steps:",
    TOTAL_STEPS - start_step
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    EFFECTIVE_BATCH_SIZE
)

print(
    "Block size:",
    BLOCK_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Minimum learning rate:",
    MIN_LEARNING_RATE
)

print(
    "Warmup steps:",
    WARMUP_STEPS
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "Validation interval:",
    VALIDATION_INTERVAL
)

print()


# ============================================================
# TRAINING LOOP
# ============================================================

model.train()


for step in range(
    start_step,
    TOTAL_STEPS
):

    optimizer.zero_grad(
        set_to_none=True
    )


    accumulated_loss = 0.0


    # ========================================================
    # GRADIENT ACCUMULATION
    # ========================================================

    for accumulation_step in range(
        GRADIENT_ACCUMULATION_STEPS
    ):

        x, y = get_batch(
            train_data
        )


        logits, loss = model(
            x,
            y
        )


        loss_for_backward = (
            loss
            /
            GRADIENT_ACCUMULATION_STEPS
        )


        loss_for_backward.backward()


        accumulated_loss += (
            loss.item()
        )


    # ========================================================
    # GRADIENT CLIPPING
    # ========================================================

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRADIENT_CLIP
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    learning_rate = get_learning_rate(
        step
    )


    for parameter_group in (
        optimizer.param_groups
    ):

        parameter_group[
            "lr"
        ] = learning_rate


    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    optimizer.step()


    average_train_loss = (
        accumulated_loss
        /
        GRADIENT_ACCUMULATION_STEPS
    )


    # ========================================================
    # PRINT TRAINING PROGRESS
    # ========================================================

    display_step = step + 1


    if (
        display_step % PRINT_INTERVAL == 0
    ):

        print(
            f"Step {display_step} / "
            f"{TOTAL_STEPS} | "
            f"Train Loss: "
            f"{average_train_loss:.4f} | "
            f"LR: "
            f"{learning_rate:.7f}"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    if (
        display_step % VALIDATION_INTERVAL == 0
        or display_step == TOTAL_STEPS
    ):

        validation_loss = (
            estimate_validation_loss(
                model
            )
        )


        print(
            "-" * 40
        )

        print(
            f"Step {display_step} | "
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        if (
            validation_loss
            <
            best_val_loss
        ):

            best_val_loss = (
                validation_loss
            )


            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT
            )


            print(
                "NEW BEST V4 MODEL SAVED!"
            )

            print(
                "Checkpoint:",
                BEST_CHECKPOINT
            )


        # ====================================================
        # FULL RESUME CHECKPOINT
        # ====================================================

        checkpoint = {

            "step": display_step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_val_loss":
                best_val_loss,

            "vocab_size":
                VOCAB_SIZE,

            "block_size":
                BLOCK_SIZE,

            "batch_size":
                BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,

            "learning_rate":
                learning_rate

        }


        torch.save(
            checkpoint,
            LAST_CHECKPOINT
        )


        print(
            "Resume checkpoint saved:"
        )

        print(
            LAST_CHECKPOINT
        )

        print(
            "-" * 40
        )


# ============================================================
# FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    FINAL_CHECKPOINT
)


print()
print("=" * 60)
print("SCRIPTURELM V4 TRAINING COMPLETED")
print("=" * 60)

print()

print(
    "Final step:",
    TOTAL_STEPS
)

print(
    "Best validation loss:",
    best_val_loss
)

print()

print(
    "Best model:"
)

print(
    BEST_CHECKPOINT
)

print()

print(
    "Final model:"
)

print(
    FINAL_CHECKPOINT
)

print()

print(
    "Resume checkpoint:"
)

print(
    LAST_CHECKPOINT
)

print()

print("=" * 60)
print("V4 TRAINING COMPLETE")
print("=" * 60)