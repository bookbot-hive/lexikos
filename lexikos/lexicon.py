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
from typing import Dict, List, Set, Tuple
import re

from .languages import (
    DictionarySource,
    LanguagePack,
    Pronunciation,
    PronunciationSource,
    get_language_pack,
    supported_lexicon_languages,
)
from .pronunciations import split_pronunciation_variants


_DICTIONARIES = Path(__file__).parent / "dict"


def _source_sort_key(source: PronunciationSource) -> Tuple:
    dialect = source.dialect
    if dialect is None:
        dialect_key = ("", "", "", "", ())
    else:
        dialect_key = (
            dialect.territory or "",
            dialect.macroregion or "",
            dialect.group or "",
            dialect.locality or "",
            tuple((feature.name, feature.value) for feature in dialect.features),
        )
    return (
        source.source,
        source.language,
        dialect_key,
        source.transcription,
        source.synthetic,
    )


def _load_pronunciations(
    pack: LanguagePack,
    sources: Tuple[DictionarySource, ...],
    normalize_phonemes: bool,
) -> Dict[str, List[Pronunciation]]:
    phoneme_normalizer = pack.phoneme_normalizer
    if normalize_phonemes and phoneme_normalizer is None:
        raise ValueError(
            "Language {!r} does not define a phoneme normalizer.".format(pack.id)
        )

    merged: Dict[str, Dict[str, Set[PronunciationSource]]] = {}
    for dictionary in sources:
        provenance = dictionary.pronunciation_source(pack.id)
        path = _DICTIONARIES / dictionary.path
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                word, raw_phonemes = line.rstrip("\r\n").split("\t", 1)
                for phonemes in split_pronunciation_variants(raw_phonemes):
                    phonemes = re.sub(r"\s+", " ", phonemes.replace(".", " ")).strip()
                    if normalize_phonemes:
                        phonemes = phoneme_normalizer(phonemes)
                    pronunciations = merged.setdefault(word.lower(), {})
                    pronunciations.setdefault(phonemes, set()).add(provenance)

    return {
        word: [
            Pronunciation(
                ipa=ipa,
                sources=tuple(sorted(provenance, key=_source_sort_key)),
            )
            for ipa, provenance in sorted(pronunciations.items())
        ]
        for word, pronunciations in merged.items()
    }


class Lexicon(UserDict):
    """A language-specific dictionary of IPA pronunciations and provenance."""

    def __init__(
        self,
        lang: str,
        *,
        normalize_phonemes: bool = False,
        include_synthetic: bool = False,
    ):
        pack = get_language_pack(lang)
        sources = tuple(
            source
            for source in pack.dictionaries
            if include_synthetic or not source.synthetic
        )
        mapping = _load_pronunciations(pack, sources, normalize_phonemes)
        super().__init__(mapping)

    @classmethod
    def supported_languages(cls) -> Tuple[str, ...]:
        return supported_lexicon_languages()


if __name__ == "__main__":
    lexicon = Lexicon("en-us")
    print(lexicon["added"])
    print(lexicon["runner"])
    print(lexicon["water"])
