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


class Lexicon(UserDict):
    def __init__(self, dictionaries_dir: Union[Path, str]):
        if isinstance(dictionaries_dir, str):
            dictionaries_dir = Path(dictionaries_dir)

        self._files = list(dictionaries_dir.rglob("*/*.tsv"))
        mapping: Dict[str, Set[str]] = self._load_files(self._files)
        super().__init__(mapping)

    def _load_files(self, files: List[Path]):
        lex = self._merge_dicts([self._parse_tsv(file) for file in files])
        return lex

    def _parse_tsv(self, file: Union[Path, str]) -> Dict[str, Set[str]]:
        lex = {}
        with open(file, "r") as f:
            for line in f.readlines():
                word, phonemes = line.strip().split("\t")
                phonemes = [phonemes.replace(" . ", " ")]
                word = word.lower()
                if word in lex:
                    lex[word] = lex[word].union(phonemes)
                else:
                    lex[word] = set(phonemes)
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


if __name__ == "__main__":
    lexicon = Lexicon("dict")
    print(lexicon["added"])
    print(lexicon["runner"])
    print(lexicon["water"])
