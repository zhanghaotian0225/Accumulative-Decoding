"""
Evaluation script for the MM-Vet benchmark.

MM-Vet tests integrated multimodal capabilities across six skills:
    Rec (Recognition), OCR, Know (Knowledge), Gen (Language Generation),
    Spat (Spatial Awareness), Math.

Expected data layout::

    <data_path>/
        images/  *.jpg / *.png
        mm-vet.json       # official annotation file

Scoring: MM-Vet uses GPT-4 as a judge to score open-ended responses
(0-1 per question).  This script generates and saves model responses;
submit the output JSON to the official evaluator or use your own GPT-4
scoring pipeline.

Official repo: https://github.com/yuweihao/MM-Vet
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


def run_mmvet(args):
    annotation_path = os.path.join(args.data_path, "mm-vet.json")
    image_dir = os.path.join(args.data_path, "images")

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict: question_id -> {question, answer, capability, ...}

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name
    )
    model.eval()

    token_embs = get_token_embeddings(model)
    placeholder_v0 = torch.zeros(token_embs.shape[1], device=model.device)
    ad_processor = AccumulativeDecodingProcessor(
        placeholder_v0, token_embs, args.alpha, args.beta, args.gamma
    )

    results = {}

    for qid, item in tqdm(data.items(), desc="MM-Vet"):
        img_name = item["imagename"]
        question = item["question"]
        capability = item.get("capability", [])

        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor[0].unsqueeze(0).to(model.device, dtype=torch.float16)

        v0 = get_llava_visual_embedding(model, image_tensor)
        ad_processor.update_visual_embedding(v0)

        prompt = DEFAULT_IMAGE_TOKEN + f"\n{question}"
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                logits_processor=[ad_processor],
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
            )

        # Decode only the newly generated tokens (exclude the input prompt)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        results[qid] = {
            "question": question,
            "answer": item.get("answer", ""),
            "model_response": response,
            "capability": capability,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mmvet_responses.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Responses saved to {out_path}")
    print("Submit this file to the MM-Vet GPT-4 evaluator for final scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="results/mmvet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.8)
    args = parser.parse_args()
    run_mmvet(args)
