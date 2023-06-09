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

from collections import UserDict
from pathlib import Path
from typing import Any, Dict, List, Set, Union
import os


class Lexicon(UserDict):
    def __init__(
        self,
        normalize_phonemes: bool = False,
        include_synthetic: bool = False,
    ):
        dictionaries_dir = Path(os.path.join(os.path.dirname(__file__), "dict"))
        files = list(dictionaries_dir.rglob("*/*.tsv"))
        synthetic_files = list(dictionaries_dir.rglob("synthetic/*.tsv"))
        if not include_synthetic:
            files = filter(lambda x: x not in synthetic_files, files)
        dicts = [self._parse_tsv(file, normalize_phonemes) for file in files]
        mapping: Dict[str, Set[str]] = self._merge_dicts(dicts)
        super().__init__(mapping)

    def _parse_tsv(
        self, file: Union[Path, str], normalize_phonemes: bool
    ) -> Dict[str, Set[str]]:
        lex = {}
        with open(file, "r") as f:
            for line in f.readlines():
                word, phonemes = line.strip().split("\t")
                phonemes = phonemes.replace(" . ", " ")
                if normalize_phonemes:
                    phonemes = self._normalize_phonemes(phonemes)
                word = word.lower()
                lex[word] = lex.get(word, set()) | set([phonemes])
        return lex

    def _merge_dicts(self, dicts: List[Dict[Any, Set]]):
        output_dict = dicts[0]
        for d in dicts[1:]:
            for k, v in d.items():
                if k in output_dict:
                    output_dict[k] = output_dict[k].union(v)
                else:
                    output_dict[k] = v
        return output_dict

    @staticmethod
    def _normalize_phonemes(phonemes: str) -> str:
        """
        Modified from: [Michael McAuliffe](https://memcauliffe.com/speaker-dictionaries-and-multilingual-ipa.html#multilingual-ipa-mode)
        """
        diacritics = ["ː", "ˑ", "̆", "̯", "͡", "‿", "͜", "̩", "ˈ", "ˌ"]
        digraphs = ["o ʊ", "e ɪ", "a ʊ", "ɑ ɪ", "a ɪ", "ɔ ɪ"]
        for d in diacritics:
            phonemes = phonemes.replace(d, "")
        for dg in digraphs:
            phonemes = phonemes.replace(dg, dg.replace(" ", ""))
        phonemes = phonemes.strip()
        return phonemes


if __name__ == "__main__":
    lexicon = Lexicon()
    print(lexicon["added"])
    print(lexicon["runner"])
    print(lexicon["water"])
