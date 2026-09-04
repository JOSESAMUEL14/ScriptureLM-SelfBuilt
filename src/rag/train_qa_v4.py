import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# PATHS
# ============================================================

DATASET_FILE = (
    BASE_DIR
    / "data"
    / "rag"
    / "qa_dataset_v4.json"
)

TOKENIZER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "tokenizer_v4.json"
)

# Original pretrained V4 model
BEST_MODEL = (
    BASE_DIR
    / "data"
    / "processed"
    / "scripturelm_v4_best.pth"
)

# Best QA model
OUTPUT_MODEL = (
    BASE_DIR
    / "data"
    / "processed"
    / "scripturelm_v4_qa.pth"
)

# Resume checkpoint
RESUME_MODEL = (
    BASE_DIR
    / "data"
    / "processed"
    / "scripturelm_v4_qa_resume.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

BLOCK_SIZE = 256

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 16

# Total number of QA epochs
EPOCHS = 3

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

MAX_TRAIN_SAMPLES = None

# True = continue from existing QA checkpoint
RESUME_TRAINING = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TOKENIZER
# ============================================================

class BPETokenizer:

    def __init__(self, tokenizer_file):

        with open(
            tokenizer_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        self.tokens = data["tokens"]

        self.merges = data["merges"]

        self.vocab_size = data["vocab_size"]

        self.token_to_id = {
            token: index
            for index, token in enumerate(
                self.tokens
            )
        }

        self.id_to_token = {
            index: token
            for index, token in enumerate(
                self.tokens
            )
        }

        self.merge_ranks = {}

        for rank, merge in enumerate(
            self.merges
        ):

            if isinstance(merge, list):

                pair = tuple(merge)

            else:

                parts = merge.split()

                pair = tuple(parts)

            if len(pair) == 2:

                self.merge_ranks[pair] = rank


    def merge_pair(
        self,
        tokens,
        pair
    ):

        merged = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):

                merged.append(
                    tokens[i] + tokens[i + 1]
                )

                i += 2

            else:

                merged.append(
                    tokens[i]
                )

                i += 1

        return merged


    def encode(
        self,
        text
    ):

        if not text:

            return []

        chars = list(text)

        while len(chars) > 1:

            pairs = [
                (
                    chars[i],
                    chars[i + 1]
                )
                for i in range(
                    len(chars) - 1
                )
            ]

            valid_pairs = [
                pair
                for pair in pairs
                if pair in self.merge_ranks
            ]

            if not valid_pairs:

                break

            best_pair = min(
                valid_pairs,
                key=lambda pair:
                    self.merge_ranks[pair]
            )

            chars = self.merge_pair(
                chars,
                best_pair
            )

        ids = []

        for token in chars:

            if token in self.token_to_id:

                ids.append(
                    self.token_to_id[token]
                )

            else:

                ids.append(0)

        return ids


    def decode(
        self,
        ids
    ):

        return "".join(
            self.id_to_token.get(
                int(token_id),
                ""
            )
            for token_id in ids
        )


# ============================================================
# DATASET
# ============================================================

class ScriptureQADataset(Dataset):

    def __init__(
        self,
        examples,
        tokenizer
    ):

        self.examples = []

        self.tokenizer = tokenizer

        print(
            "Encoding QA examples..."
        )

        for number, example in enumerate(
            examples,
            1
        ):

            question = example[
                "question"
            ]

            context = example[
                "context"
            ]

            answer = example[
                "answer"
            ]


            # ------------------------------------------------
            # EXACT QA PROMPT
            # ------------------------------------------------

            prompt = (
                "Question: "
                + question
                + "\n\n"
                + "Bible Context:\n"
                + context
                + "\n\n"
                + "Answer: "
            )


            # ------------------------------------------------
            # TOKENIZE
            # ------------------------------------------------

            prompt_ids = tokenizer.encode(
                prompt
            )

            answer_ids = tokenizer.encode(
                answer
            )


            if not answer_ids:

                continue


            # ------------------------------------------------
            # RESERVE SPACE FOR ANSWER
            # ------------------------------------------------

            max_prompt_length = (
                BLOCK_SIZE
                - len(answer_ids)
                - 1
            )


            if max_prompt_length <= 0:

                continue


            # ------------------------------------------------
            # TRUNCATE PROMPT
            # ------------------------------------------------

            prompt_ids = prompt_ids[
                -max_prompt_length:
            ]


            # ------------------------------------------------
            # COMPLETE SEQUENCE
            # ------------------------------------------------

            input_ids = (
                prompt_ids
                +
                answer_ids
            )


            if len(input_ids) < 2:

                continue


            input_ids = input_ids[
                :BLOCK_SIZE
            ]


            # ------------------------------------------------
            # ANSWER START
            # ------------------------------------------------

            answer_start = len(prompt_ids)


            self.examples.append(
                {
                    "input_ids": torch.tensor(
                        input_ids,
                        dtype=torch.long
                    ),

                    "answer_start": answer_start
                }
            )


            if number % 1000 == 0:

                print(
                    f"  Encoded {number} "
                    f"/ {len(examples)}"
                )


        print(
            "Usable examples:",
            len(self.examples)
        )


    def __len__(self):

        return len(
            self.examples
        )


    def __getitem__(
        self,
        index
    ):

        item = self.examples[index]

        tokens = item["input_ids"]

        answer_start = item["answer_start"]


        # ----------------------------------------------------
        # SHIFT
        # ----------------------------------------------------

        x = tokens[:-1]

        y = tokens[1:]


        # ----------------------------------------------------
        # ANSWER-ONLY LOSS MASK
        # ----------------------------------------------------

        targets = torch.full(
            (len(y),),
            -100,
            dtype=torch.long
        )


        answer_target_start = max(
            0,
            answer_start - 1
        )


        targets[
            answer_target_start:
        ] = y[
            answer_target_start:
        ]


        # ----------------------------------------------------
        # PADDING
        # ----------------------------------------------------

        if len(x) < BLOCK_SIZE - 1:

            padding_length = (
                BLOCK_SIZE
                - 1
                - len(x)
            )

            padding = torch.zeros(
                padding_length,
                dtype=torch.long
            )

            x = torch.cat(
                [
                    x,
                    padding
                ]
            )


            target_padding = torch.full(
                (
                    padding_length,
                ),
                -100,
                dtype=torch.long
            )

            targets = torch.cat(
                [
                    targets,
                    target_padding
                ]
            )


        else:

            x = x[
                :BLOCK_SIZE - 1
            ]

            targets = targets[
                :BLOCK_SIZE - 1
            ]


        return x, targets


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()

    print(
        "Loading ScriptureLM V4..."
    )


    sys.path.insert(
        0,
        str(BASE_DIR)
    )


    from src.model_v4 import LanguageModel


    model = LanguageModel()


    # --------------------------------------------------------
    # DETERMINE CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = None


    if (
        RESUME_TRAINING
        and
        RESUME_MODEL.exists()
    ):

        checkpoint_path = RESUME_MODEL

        print(
            "Resuming from resume checkpoint..."
        )

    elif (
        RESUME_TRAINING
        and
        OUTPUT_MODEL.exists()
    ):

        checkpoint_path = OUTPUT_MODEL

        print(
            "Resuming from saved QA model..."
        )

    else:

        checkpoint_path = BEST_MODEL

        print(
            "Starting from pretrained V4 model..."
        )


    print(
        "Checkpoint:",
        checkpoint_path
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # LOAD MODEL STATE
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model.to(
        DEVICE
    )


    return model, checkpoint


# ============================================================
# ANSWER-ONLY LOSS
# ============================================================

def calculate_answer_loss(
    logits,
    targets
):

    batch_size = logits.size(0)

    vocab_size = logits.size(-1)


    logits = logits.reshape(
        batch_size * logits.size(1),
        vocab_size
    )


    targets = targets.reshape(
        batch_size * targets.size(1)
    )


    loss = nn.functional.cross_entropy(
        logits,
        targets,
        ignore_index=-100
    )


    return loss


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    batches = 0


    for x, targets in loader:

        x = x.to(
            DEVICE
        )

        targets = targets.to(
            DEVICE
        )


        logits, _ = model(
            x
        )


        loss = calculate_answer_loss(
            logits,
            targets
        )


        if torch.isfinite(loss):

            total_loss += loss.item()

            batches += 1


    model.train()


    if batches == 0:

        return 0.0


    return (
        total_loss
        /
        batches
    )


# ============================================================
# SAVE RESUME CHECKPOINT
# ============================================================

def save_resume_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
    validation_loss,
    best_validation_loss
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "global_step":
                global_step,

            "validation_loss":
                validation_loss,

            "best_validation_loss":
                best_validation_loss
        },
        RESUME_MODEL
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    print(
        "=" * 70
    )

    print(
        "SCRIPTURELM V4 QA FINE-TUNING"
    )

    print(
        "ANSWER-ONLY LOSS + RESUME"
    )

    print(
        "=" * 70
    )


    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Dataset:",
        DATASET_FILE
    )


    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = BPETokenizer(
        TOKENIZER_FILE
    )


    print(
        "Tokenizer vocabulary:",
        tokenizer.vocab_size
    )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        dataset = json.load(f)


    train_examples = dataset[
        "train"
    ]

    validation_examples = dataset[
        "validation"
    ]


    print()

    print(
        "Train examples:",
        len(train_examples)
    )

    print(
        "Validation examples:",
        len(validation_examples)
    )


    # ========================================================
    # OPTIONAL TRAIN LIMIT
    # ========================================================

    if MAX_TRAIN_SAMPLES is not None:

        train_examples = train_examples[
            :MAX_TRAIN_SAMPLES
        ]


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = ScriptureQADataset(
        train_examples,
        tokenizer
    )

    validation_dataset = ScriptureQADataset(
        validation_examples,
        tokenizer
    )


    # ========================================================
    # LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print()

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )


    # ========================================================
    # MODEL
    # ========================================================

    model, checkpoint = load_model()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # RESUME INFORMATION
    # ========================================================

    start_epoch = 1

    global_step = 0

    best_validation_loss = float(
        "inf"
    )


    if (
        RESUME_TRAINING
        and
        isinstance(
            checkpoint,
            dict
        )
        and
        "epoch" in checkpoint
    ):

        completed_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = (
            completed_epoch + 1
        )


        global_step = int(
            checkpoint.get(
                "global_step",
                0
            )
        )


        best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get(
                    "validation_loss",
                    float("inf")
                )
            )
        )


        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )


        print()

        print(
            "RESUME INFORMATION"
        )

        print(
            "-" * 70
        )

        print(
            "Completed epoch:",
            completed_epoch
        )

        print(
            "Starting epoch:",
            start_epoch
        )

        print(
            "Global step:",
            global_step
        )

        print(
            "Best validation loss:",
            best_validation_loss
        )

        print(
            "-" * 70
        )


    # ========================================================
    # ALREADY COMPLETE?
    # ========================================================

    if start_epoch > EPOCHS:

        print()

        print(
            "QA training is already complete."
        )

        print(
            f"Completed epochs: {EPOCHS}"
        )

        print(
            "Model:",
            OUTPUT_MODEL
        )

        return


    # ========================================================
    # TRAINING
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        f"STARTING QA TRAINING FROM EPOCH {start_epoch}"
    )

    print(
        "=" * 70
    )


    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad()


        for batch_number, (
            x,
            targets
        ) in enumerate(
            train_loader,
            1
        ):

            x = x.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            logits, _ = model(
                x
            )


            # ------------------------------------------------
            # ANSWER-ONLY LOSS
            # ------------------------------------------------

            loss = calculate_answer_loss(
                logits,
                targets
            )


            if not torch.isfinite(loss):

                print(
                    "WARNING: Non-finite loss detected."
                )

                optimizer.zero_grad()

                continue


            # ------------------------------------------------
            # GRADIENT ACCUMULATION
            # ------------------------------------------------

            scaled_loss = (
                loss
                /
                GRADIENT_ACCUMULATION_STEPS
            )


            scaled_loss.backward()


            running_loss += loss.item()


            # ------------------------------------------------
            # OPTIMIZER STEP
            # ------------------------------------------------

            if (
                batch_number
                %
                GRADIENT_ACCUMULATION_STEPS
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )


                optimizer.step()

                optimizer.zero_grad()

                global_step += 1


                if global_step % 10 == 0:

                    print(
                        f"Epoch {epoch} | "
                        f"Batch {batch_number}/"
                        f"{len(train_loader)} | "
                        f"Step {global_step} | "
                        f"Answer Loss {loss.item():.4f}"
                    )


        # ====================================================
        # REMAINING GRADIENTS
        # ====================================================

        if (
            len(train_loader)
            %
            GRADIENT_ACCUMULATION_STEPS
            != 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            optimizer.zero_grad()

            global_step += 1


        # ====================================================
        # VALIDATION
        # ====================================================

        validation_loss = evaluate(
            model,
            validation_loader
        )


        print()

        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Validation Answer Loss: "
            f"{validation_loss:.6f}"
        )

        print(
            "-" * 70
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            validation_loss
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "global_step":
                        global_step,

                    "validation_loss":
                        validation_loss,

                    "best_validation_loss":
                        best_validation_loss
                },
                OUTPUT_MODEL
            )


            print()

            print(
                "NEW BEST QA MODEL SAVED!"
            )

            print(
                OUTPUT_MODEL
            )


        # ====================================================
        # ALWAYS SAVE RESUME CHECKPOINT
        # ====================================================

        save_resume_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            validation_loss,
            best_validation_loss
        )


        print()

        print(
            "RESUME CHECKPOINT SAVED!"
        )

        print(
            RESUME_MODEL
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "QA FINE-TUNING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Best validation answer loss:",
        best_validation_loss
    )

    print(
        "Best QA model:",
        OUTPUT_MODEL
    )

    print(
        "Resume checkpoint:",
        RESUME_MODEL
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()