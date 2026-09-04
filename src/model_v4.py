import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SCRIPTURELM V4 - TRANSFORMER LANGUAGE MODEL
# ============================================================

VOCAB_SIZE = 2048
BLOCK_SIZE = 256

N_EMBD = 256
N_HEAD = 8
N_LAYER = 6

DROPOUT = 0.1


# ============================================================
# CAUSAL SELF-ATTENTION HEAD
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        # Attention scores
        weights = (
            q @ k.transpose(-2, -1)
        ) * (
            k.shape[-1] ** -0.5
        )

        # Causal mask
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        weights = self.dropout(
            weights
        )

        v = self.value(x)

        output = weights @ v

        return output


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_size
    ):

        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(head_size)
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(
            N_EMBD,
            N_EMBD
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )


    def forward(self, x):

        output = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        output = self.projection(
            output
        )

        output = self.dropout(
            output
        )

        return output


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                N_EMBD,
                4 * N_EMBD
            ),

            nn.GELU(),

            nn.Linear(
                4 * N_EMBD,
                N_EMBD
            ),

            nn.Dropout(
                DROPOUT
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):

    def __init__(
        self,
        num_heads
    ):

        super().__init__()

        head_size = N_EMBD // num_heads

        self.ln1 = nn.LayerNorm(
            N_EMBD
        )

        self.attention = MultiHeadAttention(
            num_heads,
            head_size
        )

        self.ln2 = nn.LayerNorm(
            N_EMBD
        )

        self.feed_forward = FeedForward()


    def forward(self, x):

        # Pre-norm attention
        x = x + self.attention(
            self.ln1(x)
        )

        # Pre-norm feed-forward
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# SCRIPTURELM V4 LANGUAGE MODEL
# ============================================================

class LanguageModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Token embeddings
        self.token_embedding_table = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD
        )

        # Positional embeddings
        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(N_HEAD)
                for _ in range(N_LAYER)
            ]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(
            N_EMBD
        )

        # Language-model output head
        self.lm_head = nn.Linear(
            N_EMBD,
            VOCAB_SIZE
        )

        # Initialize weights
        self.apply(
            self._init_weights
        )


    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )


        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > BLOCK_SIZE:

            raise ValueError(
                f"Sequence length {T} "
                f"exceeds BLOCK_SIZE {BLOCK_SIZE}"
            )

        # Token embeddings
        token_embeddings = (
            self.token_embedding_table(idx)
        )

        # Position embeddings
        positions = torch.arange(
            T,
            device=idx.device
        )

        position_embeddings = (
            self.position_embedding_table(
                positions
            )
        )

        # Combine embeddings
        x = (
            token_embeddings
            +
            position_embeddings
        )

        # Transformer
        x = self.blocks(x)

        # Final normalization
        x = self.ln_f(x)

        # Vocabulary logits
        logits = self.lm_head(x)


        # ====================================================
        # LOSS
        # ====================================================

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SCRIPTURELM V4 MODEL TEST")
    print("=" * 60)

    print()

    print("Vocabulary size:", VOCAB_SIZE)
    print("Block size:", BLOCK_SIZE)
    print("Embedding size:", N_EMBD)
    print("Attention heads:", N_HEAD)
    print("Transformer layers:", N_LAYER)
    print("Dropout:", DROPOUT)

    print()

    model = LanguageModel()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Parameters:",
        parameter_count
    )

    print()

    # Test input
    test_input = torch.randint(
        0,
        VOCAB_SIZE,
        (
            2,
            BLOCK_SIZE
        )
    )

    test_target = torch.randint(
        0,
        VOCAB_SIZE,
        (
            2,
            BLOCK_SIZE
        )
    )

    print("Input shape:")
    print(test_input.shape)

    print()

    print("Target shape:")
    print(test_target.shape)

    print()

    # Forward pass
    logits, loss = model(
        test_input,
        test_target
    )

    print("Logits shape:")
    print(logits.shape)

    print()

    print("Loss:")
    print(loss.item())

    print()

    # Check expected shapes
    assert logits.shape == (
        2,
        BLOCK_SIZE,
        VOCAB_SIZE
    )

    assert loss is not None

    print("[OK] Forward pass PASSED")

    print()

    print("=" * 60)
    print("SCRIPTURELM V4 MODEL TEST COMPLETE")
    print("=" * 60)