from pathlib import Path
import argparse
import shutil

from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer, AutoQuantizationConfig

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace Hub ONNX model name."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="HuggingFace Hub quantized model ID for pushing",
    )
    parser.add_argument(
        "--architecture",
        choices=["arm64", "avx2", "avx512", "avx512_vnni", "tensorrt"],
        required=True,
        help="Quantization target architecture",
    )
    return parser.parse_args()

def main(args):
    if "/" in args.model_name:
        _, model_name = args.model_name.split("/")
    else:
        model_name = args.model_name

    save_dir = Path(f"{model_name}-quantized-{args.architecture}")

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(args.model_name)
    model_dir = ort_model.model_save_dir

    if (model_dir / "encoder_model_optimized.onnx").exists():
        encoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="encoder_model_optimized.onnx")
        decoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model_optimized.onnx")
        decoder_wp_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_with_past_model_optimized.onnx")
    else:
        encoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="encoder_model.onnx")
        decoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model.onnx")
        decoder_wp_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_with_past_model.onnx")
        
    quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]

    q_kwargs = {"is_static": False, "per_channel": False}

    if args.architecture == "arm64":
        dqconfig = AutoQuantizationConfig.arm64(**q_kwargs)
    elif args.architecture == "avx2":
        dqconfig = AutoQuantizationConfig.avx2(**q_kwargs)
    elif args.architecture == "avx512":
        dqconfig = AutoQuantizationConfig.avx512(**q_kwargs)
    elif args.architecture == "avx512_vnni":
        dqconfig = AutoQuantizationConfig.avx512_vnni(**q_kwargs)
    elif args.architecture == "tensorrt":
        dqconfig = AutoQuantizationConfig.tensorrt(**q_kwargs)

    for q in quantizer:
        q.quantize(save_dir=save_dir, quantization_config=dqconfig)

    ort_model.push_to_hub(str(save_dir), repository_id=args.hub_model_id)

    # remove local repository after finish
    shutil.rmtree(save_dir)

if __name__  == "__main__":
    args = parse_args()
    main(args)
