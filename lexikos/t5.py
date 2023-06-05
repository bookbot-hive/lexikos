from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

class T5:
    def __init__(self, model_path: str):
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text2text-generation", 
            model=onnx_model, 
            tokenizer=tokenizer, 
            max_length=200, 
            num_beams=1
        )

    def __call__(self, text: str) -> str:
        result = self.pipeline(text)
        return result[0]["generated_text"]

if __name__ == "__main__":
    t5 = T5("bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-quantized-avx512_vnni")
    print(t5("hello"))