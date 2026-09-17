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

from collections.abc import Iterator, Mapping
from contextlib import closing
import json
from typing import Dict, List, Set, Tuple

from .languages import (
    Dialect,
    DialectFeature,
    Pronunciation,
    PronunciationSource,
    get_language_pack,
    supported_lexicon_languages,
)
from .storage import open_runtime_database


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
        source.source_id,
        source.observation_ids,
    )


def _dialect(value: str) -> Dialect:
    data = json.loads(value)
    return Dialect(
        territory=data.get("territory"),
        macroregion=data.get("macroregion"),
        group=data.get("group"),
        locality=data.get("locality"),
        features=tuple(
            DialectFeature(name=feature["name"], value=feature["value"])
            for feature in data.get("features", ())
        ),
    )


def _source(row) -> PronunciationSource:
    dialect_json = row["dialect_json"]
    return PronunciationSource(
        source=row["source_name"],
        language=row["language"],
        dialect=_dialect(dialect_json) if dialect_json is not None else None,
        transcription=row["transcription"],
        synthetic=bool(row["synthetic"]),
        source_id=row["source_id"],
        observation_ids=tuple(json.loads(row["observation_ids_json"])),
        source_url=row["source_url"],
        license_id=row["license_id"],
        license_url=row["license_url"],
        source_revision=row["source_revision"],
        evidence_status=row["evidence_status"],
        source_language=row["source_language"],
        source_language_raw=row["source_language_raw"],
    )


class Lexicon(Mapping):
    """A language-specific, SQLite-backed mapping of IPA pronunciations."""

    def __init__(
        self,
        lang: str,
        *,
        normalize_phonemes: bool = False,
        include_synthetic: bool = False,
    ):
        pack = get_language_pack(lang)
        if normalize_phonemes and pack.phoneme_normalizer is None:
            raise ValueError(
                "Language {!r} does not define a phoneme normalizer.".format(lang)
            )
        self.lang = lang
        self.normalize_phonemes = normalize_phonemes
        self.include_synthetic = include_synthetic

    def __getitem__(self, word: str) -> List[Pronunciation]:
        synthetic_clause = "" if self.include_synthetic else "AND e.synthetic = 0"
        with closing(open_runtime_database()) as connection:
            rows = connection.execute(
                """
                SELECT
                    p.ipa,
                    p.phoneme_normalized_ipa,
                    e.source_id,
                    e.source_name,
                    e.language,
                    e.source_language,
                    e.source_language_raw,
                    e.observation_ids_json,
                    e.source_url,
                    e.source_revision,
                    e.license_id,
                    e.license_url,
                    e.evidence_status,
                    e.dialect_json,
                    e.transcription,
                    e.synthetic
                FROM pronunciation AS p
                JOIN evidence AS e ON e.pronunciation_id = p.id
                WHERE p.pack_id = ? AND p.normalized_word = ?
                {}
                ORDER BY p.ipa, e.id
                """.format(synthetic_clause),
                (self.lang, word),
            ).fetchall()

        if not rows:
            raise KeyError(word)

        grouped: Dict[str, Set[PronunciationSource]] = {}
        for row in rows:
            ipa = (
                row["phoneme_normalized_ipa"] if self.normalize_phonemes else row["ipa"]
            )
            if ipa is None:
                raise RuntimeError(
                    "Runtime snapshot lacks normalized IPA for {!r}.".format(self.lang)
                )
            grouped.setdefault(ipa, set()).add(_source(row))

        return [
            Pronunciation(
                ipa=ipa,
                sources=tuple(sorted(sources, key=_source_sort_key)),
            )
            for ipa, sources in sorted(grouped.items())
        ]

    def __iter__(self) -> Iterator[str]:
        synthetic_clause = "" if self.include_synthetic else "AND e.synthetic = 0"
        connection = open_runtime_database()
        try:
            rows = connection.execute(
                """
                SELECT DISTINCT p.normalized_word
                FROM pronunciation AS p
                JOIN evidence AS e ON e.pronunciation_id = p.id
                WHERE p.pack_id = ?
                {}
                ORDER BY p.normalized_word
                """.format(synthetic_clause),
                (self.lang,),
            )
            for row in rows:
                yield row["normalized_word"]
        finally:
            connection.close()

    def __len__(self) -> int:
        synthetic_clause = "" if self.include_synthetic else "AND e.synthetic = 0"
        with closing(open_runtime_database()) as connection:
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT p.normalized_word)
                FROM pronunciation AS p
                JOIN evidence AS e ON e.pronunciation_id = p.id
                WHERE p.pack_id = ?
                {}
                """.format(synthetic_clause),
                (self.lang,),
            ).fetchone()
        return int(row[0])

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        synthetic_clause = "" if self.include_synthetic else "AND e.synthetic = 0"
        with closing(open_runtime_database()) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM pronunciation AS p
                JOIN evidence AS e ON e.pronunciation_id = p.id
                WHERE p.pack_id = ? AND p.normalized_word = ?
                {}
                LIMIT 1
                """.format(synthetic_clause),
                (self.lang, word),
            ).fetchone()
        return row is not None

    @classmethod
    def supported_languages(cls) -> Tuple[str, ...]:
        return supported_lexicon_languages()
