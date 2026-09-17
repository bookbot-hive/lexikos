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

from contextlib import closing
import sqlite3
import string
from typing import List, Optional, Tuple, TYPE_CHECKING
import warnings

from nltk.tokenize import TweetTokenizer

from .languages import get_language_pack, supported_g2p_languages
from .storage import open_runtime_database

if TYPE_CHECKING:
    from .t5 import T5


class OOVWarning(UserWarning):
    """A dictionary-only G2P profile cannot pronounce an unknown token."""


class G2p:
    def __init__(
        self,
        lang: str,
        *,
        backend: Optional[str] = None,
        narrow: Optional[bool] = None,
        normalize_phonemes: bool = False,
    ):
        pack = get_language_pack(lang)
        if pack.text_normalizer is None:
            raise ValueError(
                "Language {!r} does not define a text normalizer.".format(lang)
            )
        if normalize_phonemes and pack.phoneme_normalizer is None:
            raise ValueError(
                "Language {!r} does not define a phoneme normalizer.".format(lang)
            )

        use_default = backend is None and narrow is None
        selected_backend = backend or "wikipron"
        transcription = "narrow" if narrow else "broad"
        with closing(open_runtime_database()) as connection:
            if use_default:
                profile = connection.execute(
                    """
                    SELECT id, backend, transcription, model
                    FROM g2p_profile
                    WHERE pack_id = ? AND is_default = 1
                    """,
                    (lang,),
                ).fetchone()
            else:
                profile = connection.execute(
                    """
                    SELECT id, backend, transcription, model
                    FROM g2p_profile
                    WHERE pack_id = ? AND backend = ? AND transcription = ?
                    """,
                    (lang, selected_backend, transcription),
                ).fetchone()

        if profile is None:
            choices = ", ".join(
                supported_g2p_languages(
                    None if use_default else selected_backend,
                    None if use_default else transcription,
                )
            )
            if use_default:
                requested = "default G2P"
            else:
                requested = "{} G2P for backend {!r}".format(
                    transcription, selected_backend
                )
            raise ValueError(
                "Language {!r} does not support {}. Supported languages: {}".format(
                    lang, requested, choices or "none"
                )
            )

        self.lang = lang
        self.profile_id = profile["id"]
        self.backend = profile["backend"]
        self.transcription = profile["transcription"]
        self.tokenizer = TweetTokenizer()
        self.normalize_phonemes = normalize_phonemes
        self._text_normalizer = pack.text_normalizer
        self._phoneme_normalizer = pack.phoneme_normalizer
        self._model_path = profile["model"]
        self._t5: Optional["T5"] = None

    @classmethod
    def supported_languages(
        cls,
        backend: Optional[str] = None,
        narrow: Optional[bool] = None,
    ) -> Tuple[str, ...]:
        if backend is None and narrow is None:
            return supported_g2p_languages()
        selected_backend = backend or "wikipron"
        transcription = "narrow" if narrow else "broad"
        return supported_g2p_languages(selected_backend, transcription)

    def __call__(self, text: str, keep_punctuations: bool = False) -> List[str]:
        text = self._normalize_text(text)
        tokens = self.tokenizer.tokenize(text)
        outputs = []
        with closing(open_runtime_database()) as connection:
            for token in tokens:
                phoneme, is_oov = self._phonemize(token, connection)
                if not keep_punctuations and self._is_punctuation(phoneme):
                    continue
                if self.normalize_phonemes and not is_oov:
                    phoneme = self._phoneme_normalizer(phoneme)
                outputs.append(phoneme)
        return outputs

    def _is_punctuation(self, token: str) -> bool:
        return all(character in string.punctuation for character in token)

    def _phonemize(
        self,
        token: str,
        connection: sqlite3.Connection,
    ) -> Tuple[str, bool]:
        if self._is_punctuation(token):
            return token, False

        row = connection.execute(
            """
            SELECT output_ipa
            FROM g2p_dictionary
            WHERE profile_id = ? AND lookup_word = ?
            ORDER BY ordinal DESC
            LIMIT 1
            """,
            (self.profile_id, token),
        ).fetchone()
        if row is not None:
            return row["output_ipa"], False

        if self._model_path is None:
            warnings.warn(
                "{!r} is out of vocabulary for {!r}; no G2P model is available.".format(
                    token, self.lang
                ),
                OOVWarning,
                stacklevel=3,
            )
            return token, True

        if self._t5 is None:
            self._t5 = self._get_t5_model()
        return self._t5(token), False

    def _normalize_text(self, text: str) -> str:
        text = self._text_normalizer(text)
        text = text.replace("-", " - ")
        return text.lower()

    def _get_t5_model(self) -> "T5":
        from .t5 import T5

        if self._model_path is None:
            raise RuntimeError("Cannot load a model for a dictionary-only G2P profile.")
        return T5(self._model_path)
