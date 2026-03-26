"""
Evaluation script for the MME benchmark.

MME measures multimodal perception with binary (Yes/No) questions across
14 sub-tasks: Existence, Count, Position, Color, Posters, Celebrity, Scene,
Landmark, Artwork, OCR, Commonsense Reasoning, Numerical Calculation,
Text Translation, and Code Reasoning.

Expected data layout::

    <data_path>/
        Existence/
            images/  *.jpg
            Existence.txt          # "image_name\tquestion\tground_truth"
        Count/
            images/  *.jpg
            Count.txt
        ...

Official repo: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
"""

import os
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

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

# Sub-tasks evaluated in the paper (Perception split)
PERCEPTION_TASKS = [
    "Existence", "Count", "Position", "Color",
    "Posters", "Celebrity", "Scene", "Landmark", "Artwork", "OCR",
]


def parse_yes_no(response: str) -> str:
    """Extract Yes/No from a model response (case-insensitive)."""
    resp = response.strip().lower()
    if resp.startswith("yes"):
        return "yes"
    if resp.startswith("no"):
        return "no"
    # Fallback: search anywhere in the response
    if "yes" in resp:
        return "yes"
    if "no" in resp:
        return "no"
    return "unknown"


def score_task(predictions: list) -> int:
    """
    MME scoring: +1 for each correct binary answer on a question.
    An image contributes 2 points (one positive + one negative question).
    """
    return sum(1 for p in predictions if p["pred"] == p["gt"])


def evaluate_task(task_name, task_dir, model, tokenizer, image_processor,
                  ad_processor, args):
    """Run inference on a single MME sub-task."""
    annotation_file = os.path.join(task_dir, f"{task_name}.txt")
    image_dir = os.path.join(task_dir, "images")

    if not os.path.exists(annotation_file):
        print(f"  [skip] annotation file not found: {annotation_file}")
        return None

    predictions = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = [raw_line.strip() for raw_line in f if raw_line.strip()]

    for line in tqdm(lines, desc=task_name, leave=False):
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        img_name, question, gt = parts[0], parts[1], parts[2].lower()

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor[0].unsqueeze(0).to(model.device, dtype=torch.float16)

        # Update visual embedding and reset cumulative state for this image
        v0 = get_llava_visual_embedding(model, image_tensor)
        ad_processor.update_visual_embedding(v0)

        prompt = DEFAULT_IMAGE_TOKEN + f"\n{question}\nAnswer with Yes or No."
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
        pred = parse_yes_no(raw_output)
        predictions.append({"pred": pred, "gt": gt, "image": img_name})

    return predictions


def run_mme(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name
    )
    model.eval()

    token_embs = get_token_embeddings(model)
    # Initialise with a zero v0; update_visual_embedding() will set the real
    # visual embedding before each generate() call inside evaluate_task().
    placeholder_v0 = torch.zeros(token_embs.shape[1], device=model.device)
    ad_processor = AccumulativeDecodingProcessor(
        placeholder_v0, token_embs, args.alpha, args.beta, args.gamma
    )

    total_score = 0.0
    task_scores = {}

    for task in PERCEPTION_TASKS:
        task_dir = os.path.join(args.data_path, task)
        if not os.path.isdir(task_dir):
            print(f"Task directory not found, skipping: {task_dir}")
            continue

        predictions = evaluate_task(
            task, task_dir, model, tokenizer, image_processor,
            ad_processor, args
        )
        if predictions is None:
            continue

        score = score_task(predictions)
        task_scores[task] = score
        total_score += score
        print(f"  {task:20s}: {score:.2f}")

    print(f"\n{'Total':20s}: {total_score:.2f}")

    # Save results
    out = {"task_scores": task_scores, "total": total_score}
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mme_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="results/mme")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.8)
    args = parser.parse_args()
    run_mme(args)
