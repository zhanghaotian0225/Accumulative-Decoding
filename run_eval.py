"""
Unified evaluation entry point for Accumulative Decoding.

Examples::

    python run_eval.py --benchmark mme   --model_path checkpoints/llava-v1.5-7b --data_path data/MME
    python run_eval.py --benchmark mmvet --model_path checkpoints/llava-v1.5-7b --data_path data/mm-vet
    python run_eval.py --benchmark mmmu  --model_path checkpoints/llava-v1.5-7b --data_path data/MMMU

    # Custom hyperparameters
    python run_eval.py --benchmark mme --model_path checkpoints/llava-v1.5-7b \\
        --data_path data/MME --alpha 0.5 --beta 0.3 --gamma 0.8
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Accumulative Decoding Evaluation")

    parser.add_argument(
        "--benchmark",
        choices=["mme", "mmvet", "mmmu"],
        required=True,
        help="Which benchmark to evaluate on.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the LLaVA-1.5-7B checkpoint directory.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to the benchmark data directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save result JSON files (default: results/<benchmark>).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Scale factor for cumulative grounding logits (default: 0.5).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Bias for grounding logits (default: 0.3).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Upper bound for dynamic weight lambda (default: 0.8).",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.benchmark}"

    # mme and mmvet require a local data directory; mmmu can auto-download.
    if args.benchmark in ("mme", "mmvet") and not args.data_path:
        parser.error(f"--data_path is required for --benchmark {args.benchmark}")

    if args.benchmark == "mme":
        from eval.eval_mme import run_mme
        run_mme(args)

    elif args.benchmark == "mmvet":
        from eval.eval_mmvet import run_mmvet
        run_mmvet(args)

    elif args.benchmark == "mmmu":
        from eval.eval_mmmu import run_mmmu
        run_mmmu(args)


if __name__ == "__main__":
    main()
