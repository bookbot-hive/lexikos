import json
import os
import re
import evaluate
import torch
import pandas as pd

from argparse import ArgumentParser
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from functools import partial


def log_results(result: Dataset, args):
    model_id = args.model.split("/")[-1]

    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    
    wer_result = wer.compute(
        references=result["target"], predictions=result["prediction"]
    )
    cer_result = cer.compute(
        references=result["target"], predictions=result["prediction"]
    )

    wer_result = f"WER: {round(wer_result, 2)}"
    cer_result = f"CER: {round(cer_result, 2)}"
    print(wer_result)
    print(cer_result)
    logging_dir = f"{args.logging_dir}/{model_id}/{args.dataset_name.split('/')[1]}"
    os.makedirs(logging_dir, exist_ok=True)

    with open(f"{logging_dir}/metrics.txt", "w") as f:
        f.write(wer_result)
        f.write("\n")
        f.write(cer_result)

    with open(f"{logging_dir}/log.json", "w") as f:
        data = [
            {"prediction": p, "target": t}
            for p, t in zip(result["prediction"], result["target"])
        ]
        json.dump(data, f, indent=2, ensure_ascii=True)
    

def normalize_input(batch, args):
    chars_to_ignore_regex = (
        f'[{"".join(args.chars_to_ignore)}]'
        if args.chars_to_ignore is not None
        else None
    )
    origin = batch["text"]
    
    batch["transcript"] = batch["transcript"].replace(" ", "")
    if chars_to_ignore_regex is not None:
        batch["text"] = re.sub(chars_to_ignore_regex, "", batch["text"]).strip().lower()
    else:
        batch["text"] = batch["text"].strip().lower()
        
    # if batch["text"] == "":
    #     print(f"Origin: {origin}")
    #     print(f"TARGET EMPTY")

        
    return batch


def main(args):
    dataset = load_dataset(args.dataset_name, split="test")

    p2g = pipeline("translation", model=args.model, device=0 if torch.cuda.is_available() else -1)

    def infer(batch):
        predictions = [
            out["translation_text"]
            for out in p2g(
                batch[args.source_text_column_name],
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
        ]
        batch["prediction"] = predictions
        
        batch["target"] = batch[args.target_text_column_name]
            
        return batch
    

    dataset = dataset.map(partial(normalize_input, args=args), num_proc=os.cpu_count()-2)
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
    parser.add_argument(
        "--chars_to_ignore", nargs='+', help="A list of characters to remove from the transcripts."
    )
    args = parser.parse_args()
    main(args)