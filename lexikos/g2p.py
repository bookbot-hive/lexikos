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

from typing import Dict, List
from pathlib import Path
import string
import os

from nltk.tokenize import TweetTokenizer

from normalizer import normalize_numbers
from t5 import T5

DICTIONARIES = Path(os.path.join(os.path.dirname(__file__), "dict"))


class G2p:
    def __init__(
        self, lang: str = "en-us", backend: str = "wikipron", narrow: bool = False
    ):
        self.lexicon = self._get_dictionary(lang, backend, narrow)
        self.t5 = self._get_t5_model(lang, backend, narrow)
        self.tokenizer = TweetTokenizer()

    def __call__(self, text: str, keep_punctuations: bool = False) -> List[str]:
        text = self._normalize_text(text)
        tokens = self.tokenizer.tokenize(text)
        phonemes = [self._phonemize(token) for token in tokens]
        if not keep_punctuations:
            phonemes = list(filter(lambda x: not self._is_punctuation(x), phonemes))

        return phonemes

    def _is_punctuation(self, token: str) -> bool:
        return all(t in string.punctuation for t in token)

    def _phonemize(self, token: str) -> str:
        # return punctuation as is
        if self._is_punctuation(token):
            return token

        try:
            # NOTE: this returns last pronunciation found
            phoneme = self.lexicon[token][-1]
            return phoneme
        except KeyError:
            phoneme = self.t5(token)
            return phoneme

    def _normalize_text(self, text: str) -> str:
        text = normalize_numbers(text)
        text = text.replace("-", " - ")
        text = text.lower()
        return text

    def _get_dictionary(
        self, lang: str, backend: str, narrow: bool
    ) -> Dict[str, List[str]]:
        _SUPPORTED_BACKENDS = ["wikipron"]
        _SUPPORTED_LANGUAGES = ["en-au", "en-ca", "en-in", "en-nz", "en-uk", "en-us"]

        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Backend {backend} is not supported!")

        if lang not in _SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {lang} is not supported!")

        _, region = lang.split("-")
        fname = f"eng_latn_{region}_{'narrow' if narrow else 'broad'}.tsv"
        path = DICTIONARIES / backend / fname
        lexicon = {}

        with open(path, "r") as f:
            for line in f.readlines():
                word, phonemes = line.strip().split("\t")
                phonemes = [phonemes.replace(" . ", " ")]
                word = word.lower()
                if word not in lexicon:
                    lexicon[word] = phonemes
                else:
                    lexicon[word] += phonemes

        return lexicon

    def _get_t5_model(self, lang: str, backend: str, narrow: bool) -> T5:
        _MODELS = {
            "wikipron": {
                "en-uk": {
                    "broad": "bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-quantized-avx512_vnni"
                },
                "en-us": {
                    "broad": "bookbot/onnx-byt5-small-wikipron-eng-latn-us-broad-quantized-avx512_vnni"
                },
                "en-au": {
                    "broad": "bookbot/onnx-byt5-small-wikipron-eng-latn-au-broad-quantized-avx512_vnni",
                }
            }
        }
        _SUPPORTED_BACKENDS = ["wikipron"]
        _SUPPORTED_LANGUAGES = ["en-au", "en-uk", "en-us"]

        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Backend {backend} is not supported!")

        if lang not in _SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {lang} is not supported!")
        
        if narrow:
            raise ValueError("Narrow model is not supported!")

        t5 = T5(_MODELS[backend][lang]['narrow' if narrow else 'broad'])

        return t5


if __name__ == "__main__":
    g2p = G2p(lang="en-us")
    print(g2p("Hello there! $100 is not a lot of money in 2023."))
