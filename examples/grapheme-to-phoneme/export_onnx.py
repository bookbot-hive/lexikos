from pathlib import Path
import argparse
import shutil

from optimum.onnxruntime import ORTModelForSeq2SeqLM

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace Hub model name."
    )
    parser.add_argument(
        "--hub_model_id", type=str, required=True, help="HuggingFace Hub model ID for pushing"
    )
    return parser.parse_args()

def main(args):
    if "/" in args.model_name:
        _, model_name = args.model_name.split("/")
    else:
        model_name = args.model_name

    save_dir = Path(f"onnx-{model_name}")

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(args.model_name, export=True)
    ort_model.save_pretrained(save_dir)

    ort_model.push_to_hub(str(save_dir), repository_id=args.hub_model_id)

    # remove local repository after finish
    shutil.rmtree(save_dir)

if __name__  == "__main__":
    args = parse_args()
    main(args)
