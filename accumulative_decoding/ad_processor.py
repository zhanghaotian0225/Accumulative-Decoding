"""
Accumulative Decoding (AD) — core logits processor.

Reference:
    "Mitigating Hallucinations in Large Vision-Language Models via
     Accumulative Decoding", Zhang & Zhang, 2024.

Algorithm recap (all equation numbers match the paper):
    Eq.(1)  s(yt|I) = softmax_y [ sim(Emb(y), v0) ]        -- grounding score
    Eq.(3)  Gt = sum_{i=1}^{t} s(y_{i-1}|I)               -- cumulative score
    Eq.(4)  lGt(yt) = alpha * Gt * s(yt|I) + beta          -- grounding logits
    Eq.(5)  lambda_t = gamma * sigmoid(Avg prev scores)     -- dynamic weight
    Eq.(2)  lAD(yt) = (1-lambda_t)*lbase(yt) + lambda_t*lGt(yt)  -- final logits
"""

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class AccumulativeDecodingProcessor(LogitsProcessor):
    """
    Transformers-compatible LogitsProcessor that applies Accumulative Decoding.

    Args:
        visual_embedding (Tensor): Mean-pooled projected visual embedding v0,
            shape [hidden_dim].  Obtain via ``get_llava_visual_embedding()``.
        token_embeddings (Tensor): Full token embedding matrix of the LM,
            shape [vocab_size, hidden_dim].  Obtain via ``get_token_embeddings()``.
        alpha (float): Amplitude scale for cumulative grounding logits (default 0.5).
        beta (float): Bias for cumulative grounding logits (default 0.3).
        gamma (float): Upper bound for the dynamic weight lambda (default 0.8).

    Usage::

        processor = AccumulativeDecodingProcessor(v0, token_embs)
        output = model.generate(..., logits_processor=[processor])
        # Before the next image: update visual embedding and reset state
        processor.update_visual_embedding(new_v0)
    """

    def __init__(
        self,
        visual_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.8,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Store token embeddings for use in update_visual_embedding().
        self._token_embeddings: torch.Tensor = token_embeddings

        # Precompute per-token grounding scores s(y|I) for the whole vocabulary.
        # This is Eq.(1): softmax of cosine similarities between each token
        # embedding and the visual embedding v0.
        self._gs: torch.Tensor = self._compute_grounding_scores(visual_embedding)

        # Mutable generation state — reset between sequences.
        self._cumulative: float = 0.0
        self._prev_scores: list = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_grounding_scores(self, visual_embedding: torch.Tensor) -> torch.Tensor:
        """Compute Eq.(1): softmax of cosine similarities over the full vocabulary."""
        with torch.no_grad():
            t_norm = F.normalize(self._token_embeddings.float(), dim=-1)  # [V, D]
            v_norm = F.normalize(visual_embedding.float(), dim=-1)         # [D]
            sim = t_norm @ v_norm                                           # [V]
            return F.softmax(sim, dim=-1)                                   # [V]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset cumulative state. Call this before starting a new sequence."""
        self._cumulative = 0.0
        self._prev_scores = []
        self._step = 0

    def update_visual_embedding(self, visual_embedding: torch.Tensor) -> None:
        """
        Replace the visual embedding with a new one and reset cumulative state.

        Call this once per image, before calling ``model.generate()``.
        This is the correct way to reuse a single processor instance across
        multiple images — do NOT call ``__init__`` directly.

        Args:
            visual_embedding (Tensor): New mean-pooled visual embedding v0,
                shape [hidden_dim].  Obtain via ``get_llava_visual_embedding()``.
        """
        self._gs = self._compute_grounding_scores(visual_embedding)
        self.reset()

    # ------------------------------------------------------------------
    # LogitsProcessor interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,    # [batch, seq_len]
        scores: torch.FloatTensor,      # [batch, vocab_size]
    ) -> torch.FloatTensor:
        """
        Adjust logits at generation step t.

        Note: this implementation assumes batch_size == 1, which is standard
        for single-image inference.  For batched generation, instantiate one
        processor per sequence.
        """
        device = scores.device
        gs = self._gs.to(device=device, dtype=scores.dtype)  # [V]

        # ---- Update cumulative score from the token chosen at step t-1 ----
        # Eq.(3): Gt = sum_{i=1}^{t} s(y_{i-1}|I)
        if self._step > 0:
            last_id = int(input_ids[0, -1].item())
            score_prev = float(gs[last_id].item())
            self._prev_scores.append(score_prev)
            self._cumulative += score_prev

        Gt = self._cumulative

        # ---- Dynamic weight lambda_t  Eq.(5) ----
        if self._prev_scores:
            avg_prev = sum(self._prev_scores) / len(self._prev_scores)
            lambda_t = float(
                self.gamma
                * torch.sigmoid(torch.tensor(avg_prev, dtype=torch.float32)).item()
            )
        else:
            lambda_t = 0.0

        # ---- Grounding logits lGt  Eq.(4) ----
        # lGt(yt) = alpha * Gt * s(yt|I) + beta
        l_Gt = self.alpha * Gt * gs + self.beta   # [V]

        # ---- Final adjusted logits  Eq.(2) ----
        # lAD = (1 - lambda_t) * lbase + lambda_t * lGt
        adjusted = (1.0 - lambda_t) * scores + lambda_t * l_Gt.unsqueeze(0)  # [1, V]

        self._step += 1
        return adjusted
