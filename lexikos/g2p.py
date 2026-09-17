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

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import string

from nltk.tokenize import TweetTokenizer

from .languages import (
    DictionarySource,
    get_language_pack,
    supported_g2p_languages,
)
from .pronunciations import split_pronunciation_variants
from .utils import logger

if TYPE_CHECKING:
    from .t5 import T5


_DICTIONARIES = Path(__file__).parent / "dict"


class G2p:
    def __init__(
        self,
        lang: str,
        *,
        backend: str = "wikipron",
        narrow: Optional[bool] = None,
        normalize_phonemes: bool = False,
    ):
        pack = get_language_pack(lang)
        if narrow is None:
            logger.info(
                "Neither narrow nor broad pronunciation was specified, "
                "defaulting to broad pronunciation."
            )
            narrow = False

        transcription = "narrow" if narrow else "broad"
        profile = pack.g2p_profile(backend, transcription)
        if (
            profile is None
            or not profile.dictionary.path
            or not profile.model
            or pack.text_normalizer is None
        ):
            choices = ", ".join(supported_g2p_languages(backend, transcription))
            raise ValueError(
                "Language {!r} does not support {} G2P with backend {!r}. "
                "Supported languages: {}".format(
                    lang, transcription, backend, choices or "none"
                )
            )
        if normalize_phonemes and pack.phoneme_normalizer is None:
            raise ValueError(
                "Language {!r} does not define a phoneme normalizer.".format(lang)
            )

        self.lang = lang
        self.backend = backend
        self.transcription = transcription
        self.lexicon = self._get_dictionary(profile.dictionary)
        self.tokenizer = TweetTokenizer()
        self.normalize_phonemes = normalize_phonemes
        self._text_normalizer = pack.text_normalizer
        self._phoneme_normalizer = pack.phoneme_normalizer
        self._model_path = profile.model
        self._t5: Optional["T5"] = None

    @classmethod
    def supported_languages(
        cls, backend: str = "wikipron", narrow: bool = False
    ) -> Tuple[str, ...]:
        transcription = "narrow" if narrow else "broad"
        return supported_g2p_languages(backend, transcription)

    def __call__(self, text: str, keep_punctuations: bool = False) -> List[str]:
        text = self._normalize_text(text)
        tokens = self.tokenizer.tokenize(text)
        phonemes = [self._phonemize(token) for token in tokens]
        if not keep_punctuations:
            phonemes = [
                phoneme for phoneme in phonemes if not self._is_punctuation(phoneme)
            ]
        if self.normalize_phonemes:
            phonemes = [self._phoneme_normalizer(phoneme) for phoneme in phonemes]
        return phonemes

    def _is_punctuation(self, token: str) -> bool:
        return all(character in string.punctuation for character in token)

    def _phonemize(self, token: str) -> str:
        if self._is_punctuation(token):
            return token

        try:
            return self.lexicon[token][-1]
        except KeyError:
            if self._t5 is None:
                self._t5 = self._get_t5_model()
            return self._t5(token)

    def _normalize_text(self, text: str) -> str:
        text = self._text_normalizer(text)
        text = text.replace("-", " - ")
        return text.lower()

    def _get_dictionary(self, dictionary: DictionarySource) -> Dict[str, List[str]]:
        path = _DICTIONARIES / dictionary.path
        lexicon: Dict[str, List[str]] = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                word, phonemes = line.rstrip("\r\n").split("\t", 1)
                pronunciations = [
                    pronunciation.replace(" . ", " ")
                    for pronunciation in split_pronunciation_variants(phonemes)
                ]
                word = word.lower()
                lexicon.setdefault(word, []).extend(pronunciations)
        return lexicon

    def _get_t5_model(self) -> "T5":
        from .t5 import T5

        return T5(self._model_path)


if __name__ == "__main__":
    g2p = G2p("en-us", normalize_phonemes=True)
    print(g2p("Hello there! $100 is not a lot of money in 2023."))
