# Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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