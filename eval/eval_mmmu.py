"""
Evaluation script for the MMMU benchmark (validation split).

MMMU contains college-level multiple-choice questions across 30 disciplines.
Each question has one image and up to 5 answer options (A/B/C/D/E).

Data is loaded via HuggingFace datasets::

    datasets.load_dataset("MMMU/MMMU", split="validation")

Or from a local directory with the same schema::

    datasets.load_dataset("parquet", data_dir=<data_path>, split="validation")

The MMMU HuggingFace dataset schema:
    - question   : str
    - options    : list[str]   (plain text, no letter prefix)
    - answer     : str         ("A", "B", "C", "D", or "E")
    - image_1 .. image_7 : PIL.Image or dict{"bytes": bytes} or None

Official repo: https://github.com/MMMU-Benchmark/MMMU
"""

import os
import re
import json
import argparse
from io import BytesIO

import torch
from PIL import Image
from tqdm import tqdm

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from accumulative_decoding import (
    AccumulativeDecodingProcessor,
    get_llava_visual_embedding,
    get_token_embeddings,
)


def build_option_string(options: list) -> str:
    """
    Format MMMU answer options for the prompt.

    The MMMU HuggingFace dataset stores options as a plain list of strings
    (e.g. ["Paris", "London", "Berlin", "Madrid"]).  We prepend letter labels.
    """
    labels = ["A", "B", "C", "D", "E"]
    lines = []
    for i, text in enumerate(options):
        if i >= len(labels):
            break
        lines.append(f"({labels[i]}) {text}")
    return "\n".join(lines)


def extract_choice(response: str) -> str:
    """
    Extract the single letter answer (A-E) from the model response.

    Returns an empty string when no valid choice is found, so that
    these cases are counted as incorrect rather than boosting accuracy
    with an arbitrary default.
    """
    clean = response.strip()
    # Common patterns: "A", "(A)", "A.", "Answer: A", "The answer is A"
    match = re.search(r"(?:^|[\s(])([A-E])(?:$|[\s).:])", clean)
    if match:
        return match.group(1)
    # Last resort: bare single letter in a very short response
    if len(clean) <= 3 and clean.upper() in "ABCDE":
        return clean.upper()
    return ""


def load_mmmu_dataset(data_path: str):
    """Load MMMU validation split from HuggingFace or local parquet files."""
    if hf_datasets is None:
        raise ImportError("Install `datasets` package: pip install datasets")

    if data_path and os.path.isdir(data_path):
        return hf_datasets.load_dataset("parquet", data_dir=data_path, split="validation")

    print("Downloading MMMU from HuggingFace Hub...")
    return hf_datasets.load_dataset("MMMU/MMMU", split="validation")


def run_mmmu(args):
    dataset = load_mmmu_dataset(args.data_path)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name
    )
    model.eval()

    token_embs = get_token_embeddings(model)
    # Initialise with a zero v0; update_visual_embedding() sets the real one
    # before each generate() call.
    placeholder_v0 = torch.zeros(token_embs.shape[1], device=model.device)
    ad_processor = AccumulativeDecodingProcessor(
        placeholder_v0, token_embs, args.alpha, args.beta, args.gamma
    )

    correct = 0
    total = 0
    results = []

    for item in tqdm(dataset, desc="MMMU"):
        # Each item may have up to 7 image slots; use the first non-null one.
        pil_image = None
        for img_key in [f"image_{i}" for i in range(1, 8)]:
            raw = item.get(img_key)
            if raw is None:
                continue
            if isinstance(raw, dict) and "bytes" in raw and raw["bytes"] is not None:
                pil_image = Image.open(BytesIO(raw["bytes"])).convert("RGB")
                break
            elif isinstance(raw, Image.Image):
                pil_image = raw.convert("RGB")
                break

        if pil_image is None:
            continue

        image_tensor = process_images([pil_image], image_processor, model.config)
        image_tensor = image_tensor[0].unsqueeze(0).to(model.device, dtype=torch.float16)

        v0 = get_llava_visual_embedding(model, image_tensor)
        ad_processor.update_visual_embedding(v0)

        question = item["question"]
        options_str = build_option_string(item.get("options", []))
        gt = item["answer"].strip().upper()

        prompt = (
            DEFAULT_IMAGE_TOKEN
            + f"\n{question}\n{options_str}\n"
            "Answer with the option letter from the given choices directly."
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                logits_processor=[ad_processor],
                max_new_tokens=10,
                temperature=0,
                do_sample=False,
            )

        # Decode only the newly generated tokens (exclude the input prompt)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_choice(raw_output)

        is_correct = pred == gt
        correct += int(is_correct)
        total += 1
        results.append({
            "id": item.get("id", total),
            "question": question,
            "gt": gt,
            "pred": pred,
            "correct": is_correct,
        })

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"\nMMMU Accuracy: {accuracy:.2f}% ({correct}/{total})")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mmmu_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": accuracy, "correct": correct, "total": total, "results": results},
            f, indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--data_path", default=None,
        help="Local parquet directory, or omit to auto-download from HuggingFace Hub."
    )
    parser.add_argument("--output_dir", default="results/mmmu")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.8)
    args = parser.parse_args()
    run_mmmu(args)
