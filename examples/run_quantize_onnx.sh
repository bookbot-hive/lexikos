python3 quantize_onnx.py \
    --model_name bookbot/onnx-byt5-small-wikipron-eng-latn-nz-broad \
    --hub_model_id bookbot/onnx-byt5-small-wikipron-eng-latn-nz-broad-quantized-avx512_vnni \
    --architecture avx512_vnni