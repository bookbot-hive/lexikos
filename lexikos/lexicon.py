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
import re


class Lexicon(UserDict):
    def __init__(
        self, normalize_phonemes: bool = False, include_synthetic: bool = False, standardize_wikipron: bool = False
    ):
        dictionaries_dir = Path(os.path.join(os.path.dirname(__file__), "dict"))
        files = list(dictionaries_dir.rglob("*/*.tsv"))
        synthetic_files = list(dictionaries_dir.rglob("synthetic/*.tsv"))
        wikipron_files = list(dictionaries_dir.rglob("wikipron/*.tsv"))
        if not include_synthetic:
            files = filter(lambda x: x not in synthetic_files, files)

        if not standardize_wikipron:
            dicts = [self._parse_tsv(file, normalize_phonemes) for file in files]
        else:
            dicts = [self._parse_tsv(file, normalize_phonemes) for file in files if file not in wikipron_files]
            wikipron = [self._parse_tsv(file, normalize_phonemes, standardize_wikipron) for file in wikipron_files]
            dicts += wikipron

        mapping: Dict[str, Set[str]] = self._merge_dicts(dicts)
        super().__init__(mapping)

    def _parse_tsv(
        self, file: Union[Path, str], normalize_phonemes: bool, standardize_wikipron: bool = False
    ) -> Dict[str, Set[str]]:
        lex = {}
        with open(file, "r") as f:
            for line in f.readlines():
                word, _phonemes = line.strip().split("\t")
                word = word.lower()
                for phonemes in _phonemes.split(" ~ "):
                    phonemes = phonemes.replace(".", " ")
                    phonemes = re.sub("\s+", " ", phonemes)
                    if standardize_wikipron:
                        phonemes = self._standardize_wikipron_phonemes(phonemes)
                    elif normalize_phonemes:
                        phonemes = self._normalize_phonemes(phonemes)
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

    @staticmethod
    def _standardize_wikipron_phonemes(phonemes: str) -> str:
        """
        Standardize pronunciation phonemes from Wiktionary.
        Inspired by [Michael McAuliffe](https://mmcauliffe.medium.com/creating-english-ipa-dictionary-using-montreal-forced-aligner-2-0-242415dfee32).
        """
        diacritics = ["ː", "ˑ", "̆", "̯", "͡", "‿", "͜", "̩", "ˈ", "ˌ", "↓"]
        digraphs = {
            "a i": "aɪ",
            "a j": "aɪ",
            "a u": "aʊ",
            "a ɪ": "aɪ",
            "a ɪ̯": "aɪ",
            "a ʊ": "aʊ",
            "a ʊ̯": "aʊ",
            "d ʒ": "dʒ",
            "e i": "eɪ",
            "e ɪ": "eɪ",
            "e ɪ̯": "eɪ",
            "e ɪ̪": "eɪ",
            "o i": "ɔɪ",
            "o u": "oʊ",
            "o w": "oʊ",
            "o ɪ": "ɔɪ",
            "o ʊ": "oʊ",
            "o ʊ̯": "oʊ",
            "t ʃ": "tʃ",
            "ɑ ɪ": "aɪ",
            "ɔ i": "ɔɪ",
            "ɔ ɪ": "ɔɪ",
            "ɔ ɪ̯": "ɔɪ",
        }
        consonants = {
            "pʰ": "p",
            "b̥": "b",
            "tʰ": "t",
            "d̥": "d",
            "tʃʰ": "tʃ",
            "d̥ʒ̊": "dʒ",
            "kʰ": "k",
            "ɡ̊": "ɡ",
            "ɸ": "f",
            "β": "v",
            "v̥": "v",
            "t̪": "θ",
            "ð̥": "ð",
            "d̪": "ð",
            "z̥": "z",
            "ʒ̊": "ʒ",
            "ɦ": "h",
            "ç": "h",
            "x": "h",
            "χ": "h",
            "ɱ": "m",
            "ɫ": "l",
            "l̥": "l",
            "ɫ̥": "l",
            "ɤ": "l",
            "ɹʷ": "ɹ",
            "r": "ɹ",
            "ɻ": "ɹ",
            "ɹ̥ʷ": "ɹ",
            "ɹ̥": "ɹ",
            "ɾ̥": "ɹ",
            "ɻ̊": "ɹ",
            "ʍ": "w",
            "h w": "w",
            "ɜ ɹ": "ɚ",
        }
        vowels = {
            "ɐ": "ʌ",
            "ɒ": "ɔ",
            "ɜ": "ə",
            "ɵ": "oʊ",
            "ɘ": "ə",
        }
        leftover_vowels = {
            "a": "æ",
            "o": "ɔ",
            "e": "ɛ",
        }
        for i, j in digraphs.items():
            phonemes = phonemes.replace(i, j)
        for d in diacritics:
            phonemes = phonemes.replace(d, "")
        for i, j in consonants.items():
            phonemes = phonemes.replace(i, j)
        for i, j in vowels.items():
            phonemes = phonemes.replace(i, j)
        for i, j in leftover_vowels.items():
            phonemes = " ".join([j if p == i else p for p in phonemes.split()])
        phonemes = phonemes.strip()
        phonemes = re.sub("\s+", " ", phonemes)
        return phonemes


if __name__ == "__main__":
    lexicon = Lexicon()
    print(lexicon["added"])
    print(lexicon["runner"])
    print(lexicon["water"])
