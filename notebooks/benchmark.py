from time import perf_counter
import numpy as np
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from tqdm.auto import tqdm
from datetime import datetime


def measure_latency(pipe):
    latencies = []
    # warm up
    for _ in tqdm(range(10)):
        _ = pipe(payload)
    # Timed run
    for _ in tqdm(range(50)):
        start_time = perf_counter()
        result =  pipe(payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    return f"P95 latency (ms) - {time_p95_ms:.2f}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms


dt = datetime.now()
payload = "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you."
onnx_models = [
    "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad",
    "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-optimized",
    "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-quantized-arm64",
    "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-quantized-avx512_vnni",
    "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-optimized-quantized-arm64"
]

vanilla_model = "bookbot/byt5-small-wikipron-eng-latn-uk-broad"
tokenizer = AutoTokenizer.from_pretrained(vanilla_model)

vanilla_pipe = pipeline("text2text-generation", model=vanilla_model, tokenizer=tokenizer, max_length=200, num_beams=1)
vanilla_result = measure_latency(vanilla_pipe)
    
with open(f"logs/benchmark_{dt}.txt", "w") as f:
    f.write(f"{vanilla_model}: {vanilla_result[0]}\n")
    for model in onnx_models:
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model)
        onnx_pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer, max_length=200, num_beams=1)
        onnx_result = measure_latency(onnx_pipe)
        
        f.write(f"{model}: {onnx_result[0]}")
        f.write(f"Improvement: {round(vanilla_result[1] / onnx_result[1], 2)}x\n")
