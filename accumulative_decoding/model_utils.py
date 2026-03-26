"""
Utility functions for extracting visual embeddings and token embeddings
from a LLaVA-1.5 model, as required by AccumulativeDecodingProcessor.
"""

import torch


def get_llava_visual_embedding(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the projected visual embedding v0 from a LLaVA-1.5 model.

    LLaVA-1.5 pipeline:
        image  -->  CLIP ViT (vision_tower)  -->  mm_projector  -->  visual tokens

    We mean-pool the projected visual tokens to obtain a single embedding
    in the language model's hidden space.  This becomes the visual anchor v0
    used by Accumulative Decoding.

    Args:
        model: A loaded ``LlavaLlamaForCausalLM`` instance.
        image_tensor (Tensor): Preprocessed image, shape [1, C, H, W],
            already on the correct device and dtype.

    Returns:
        v0 (Tensor): Mean-pooled visual embedding, shape [lm_hidden_dim].
    """
    with torch.no_grad():
        vision_tower = model.get_model().get_vision_tower()
        # image_features: [1, num_patches, vision_dim]
        image_features = vision_tower(image_tensor)
        # projected: [1, num_patches, lm_hidden_dim]
        projected = model.get_model().mm_projector(image_features)

    # Average over patch dimension  ->  [lm_hidden_dim]
    v0 = projected.squeeze(0).mean(dim=0)
    return v0


def get_token_embeddings(model) -> torch.Tensor:
    """
    Retrieve the input token embedding matrix from a LLaVA (or any
    transformer-based) model.

    Args:
        model: A loaded transformer model.

    Returns:
        Embedding weight matrix, shape [vocab_size, hidden_dim].
        Returned as a detached tensor (no grad).
    """
    return model.get_input_embeddings().weight.detach()
