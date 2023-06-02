from pathlib import Path
import argparse
import shutil

from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTOptimizer, OptimizationConfig

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

    save_dir = Path(f"{model_name}-optimized")

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # monkeypatch for https://github.com/microsoft/onnxruntime/issues/14886
    ort_model.config.num_heads = 0
    ort_model.config.hidden_size = 0

    optimizer = ORTOptimizer.from_pretrained(ort_model)
    optimization_config = OptimizationConfig(
        optimization_level=99,
        enable_transformers_specific_optimizations=True,
        optimize_for_gpu=False,
    )
    optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)

    ort_model.push_to_hub(str(save_dir), repository_id=args.hub_model_id)

    # remove local repository after finish
    shutil.rmtree(save_dir)

if __name__  == "__main__":
    args = parse_args()
    main(args)
