from argparse import ArgumentParser
import json
import os

import evaluate
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def log_results(result: Dataset, args):
    model_id = args.model.split("/")[-1]

    wer = evaluate.load("wer")
    wer_result = wer.compute(
        references=result["target"], predictions=result["prediction"]
    )

    result_str = f"WER: {round(wer_result, 2)}"
    print(result_str)

    logging_dir = f"{args.logging_dir}/{model_id}"
    os.makedirs(logging_dir, exist_ok=True)

    with open(f"{logging_dir}/metrics.txt", "w") as f:
        f.write(result_str)

    with open(f"{logging_dir}/log.json", "w") as f:
        data = [
            {"prediction": p, "target": t}
            for p, t in zip(result["prediction"], result["target"])
        ]
        json.dump(data, f, indent=2, ensure_ascii=True)


def main(args):
    dataset = load_dataset(args.dataset_name, split="test")

    g2p = pipeline(model=args.model, device=0 if torch.cuda.is_available() else -1)

    def infer(batch):
        predictions = [
            out["generated_text"]
            for out in g2p(
                batch[args.source_text_column_name],
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
        ]
        batch["prediction"] = predictions
        batch["target"] = batch[args.target_text_column_name]
        return batch

    result = dataset.map(infer, batched=True, batch_size=args.batch_size)
    log_results(result, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace translation model checkpoint.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--source_text_column_name",
        type=str,
        required=True,
        help="Source text column name in dataset.",
    )
    parser.add_argument(
        "--target_text_column_name",
        type=str,
        required=True,
        help="Target text column name in dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="Maximum length of the sequence to be generated.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. 1 means no beam search.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs", help="Path to logging directory."
    )
    args = parser.parse_args()
    main(args)
