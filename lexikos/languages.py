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

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .normalizer import (
    normalize_english_phonemes,
    normalize_numbers,
    normalize_spanish_text,
)


Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class DialectFeature:
    """One explicitly sourced feature of a dialect."""

    name: str
    value: str


@dataclass(frozen=True)
class Dialect:
    """Geographic and phonological dialect metadata supplied by a source."""

    territory: Optional[str] = None
    macroregion: Optional[str] = None
    group: Optional[str] = None
    locality: Optional[str] = None
    features: Tuple[DialectFeature, ...] = ()


@dataclass(frozen=True)
class PronunciationSource:
    """Immutable provenance for one runtime pronunciation."""

    source: str
    language: str
    dialect: Optional[Dialect]
    transcription: str
    synthetic: bool
    source_id: str
    observation_ids: Tuple[str, ...]
    source_url: Optional[str]
    license_id: Optional[str]
    license_url: Optional[str]
    source_revision: str
    evidence_status: str
    source_language: Optional[str]
    source_language_raw: Optional[str]


@dataclass(frozen=True)
class Pronunciation:
    """A unique IPA pronunciation and every source that provides it."""

    ipa: str
    sources: Tuple[PronunciationSource, ...]


@dataclass(frozen=True)
class DictionarySource:
    """Build-time source declaration for one legacy dictionary artifact."""

    path: str
    source: str
    dialect: Optional[Dialect] = None
    transcription: str = "unspecified"
    synthetic: bool = False


@dataclass(frozen=True)
class G2pProfile:
    id: str
    backend: str
    transcription: str
    dictionary: DictionarySource
    model: Optional[str]


@dataclass(frozen=True)
class LanguagePack:
    id: str
    display_name: str
    base_language: str
    territory: Optional[str]
    macroregion: Optional[str]
    dictionaries: Tuple[DictionarySource, ...]
    g2p_profiles: Tuple[G2pProfile, ...]
    default_g2p_profile_id: str
    text_normalizer: Optional[Normalizer]
    phoneme_normalizer: Optional[Normalizer]

    def default_g2p_profile(self) -> G2pProfile:
        for profile in self.g2p_profiles:
            if profile.id == self.default_g2p_profile_id:
                return profile
        raise ValueError(
            "Language {!r} has no valid default G2P profile.".format(self.id)
        )

    def g2p_profile(self, backend: str, transcription: str) -> Optional[G2pProfile]:
        for profile in self.g2p_profiles:
            if profile.backend == backend and profile.transcription == transcription:
                return profile
        return None


_US = Dialect(territory="US", macroregion="north-america", group="american")
_UK = Dialect(territory="GB", macroregion="europe", group="british")
_AU = Dialect(territory="AU", macroregion="oceania", group="australian")
_NZ = Dialect(territory="NZ", macroregion="oceania", group="new-zealand")
_CA = Dialect(territory="CA", macroregion="north-america", group="canadian")
_IN = Dialect(territory="IN", macroregion="south-asia", group="indian")
_ES = Dialect(territory="ES", macroregion="europe", group="peninsular")
_LATIN_AMERICA = Dialect(macroregion="latin-america", group="latin-american")
_MX = Dialect(territory="MX", macroregion="latin-america", group="mexican")
_ID_L2 = Dialect(territory="ID", macroregion="southeast-asia", group="indonesian-l2")
_VN_L2 = Dialect(territory="VN", macroregion="southeast-asia", group="vietnamese-l2")
_CN_L2 = Dialect(territory="CN", macroregion="east-asia", group="mandarin-l2")
_EG_L2 = Dialect(territory="EG", macroregion="north-africa", group="arabic-l2")
_ES_L2 = Dialect(territory="ES", macroregion="europe", group="spanish-l2")
_IN_L2 = Dialect(territory="IN", macroregion="south-asia", group="indian-l2")
_KR_L2 = Dialect(territory="KR", macroregion="east-asia", group="korean-l2")


def _dictionary(
    path: str,
    source: str,
    dialect: Optional[Dialect] = None,
    transcription: str = "unspecified",
    synthetic: bool = False,
) -> DictionarySource:
    return DictionarySource(
        path=path,
        source=source,
        dialect=dialect,
        transcription=transcription,
        synthetic=synthetic,
    )


_ES_CHARSIU = _dictionary("charsiu/spa.tsv", "charsiu-g2p", _ES, "phonetic")
_ES_LATIN_CHARSIU = _dictionary(
    "charsiu/spa-latin.tsv", "charsiu-g2p", _LATIN_AMERICA, "phonetic"
)
_ES_MX_CHARSIU = _dictionary("charsiu/spa-me.tsv", "charsiu-g2p", _MX, "phonetic")
_ES_WIKIPRON_CA_BROAD = _dictionary(
    "wikipron/spa_latn_ca_broad.tsv", "wikipron", _ES, "broad"
)
_ES_WIKIPRON_CA_NARROW = _dictionary(
    "wikipron/spa_latn_ca_narrow.tsv", "wikipron", _ES, "narrow"
)
_ES_WIKIPRON_LA_BROAD = _dictionary(
    "wikipron/spa_latn_la_broad.tsv", "wikipron", _LATIN_AMERICA, "broad"
)
_ES_WIKIPRON_LA_NARROW = _dictionary(
    "wikipron/spa_latn_la_narrow.tsv", "wikipron", _LATIN_AMERICA, "narrow"
)


_EN_WIKIPRON = _dictionary("wikipron/eng_latn.tsv", "wikipron")
_EN_US_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_us_broad.tsv", "wikipron", _US, "broad"
)
_EN_US_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_us_narrow.tsv", "wikipron", _US, "narrow"
)
_EN_UK_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_uk_broad.tsv", "wikipron", _UK, "broad"
)
_EN_UK_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_uk_narrow.tsv", "wikipron", _UK, "narrow"
)
_EN_AU_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_au_broad.tsv", "wikipron", _AU, "broad"
)
_EN_AU_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_au_narrow.tsv", "wikipron", _AU, "narrow"
)
_EN_NZ_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_nz_broad.tsv", "wikipron", _NZ, "broad"
)
_EN_NZ_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_nz_narrow.tsv", "wikipron", _NZ, "narrow"
)
_EN_CA_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_ca_broad.tsv", "wikipron", _CA, "broad"
)
_EN_CA_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_ca_narrow.tsv", "wikipron", _CA, "narrow"
)
_EN_IN_WIKIPRON_BROAD = _dictionary(
    "wikipron/eng_latn_in_broad.tsv", "wikipron", _IN, "broad"
)
_EN_IN_WIKIPRON_NARROW = _dictionary(
    "wikipron/eng_latn_in_narrow.tsv", "wikipron", _IN, "narrow"
)

_ENGLISH_PACKS = (
    LanguagePack(
        id="en",
        display_name="English",
        base_language="en",
        territory=None,
        macroregion=None,
        dictionaries=(
            _EN_WIKIPRON,
            _dictionary("synthetic/wu_lexicon.tsv", "wu-lexicon", synthetic=True),
            _dictionary(
                "synthetic/librispeech-lexicon-en-id.tsv",
                "librispeech",
                _ID_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-vn_lexicon.tsv",
                "l2-arctic",
                _VN_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-cn_lexicon.tsv",
                "l2-arctic",
                _CN_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-eg_lexicon.tsv",
                "l2-arctic",
                _EG_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-es_lexicon.tsv",
                "l2-arctic",
                _ES_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-in_lexicon.tsv",
                "l2-arctic",
                _IN_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/l2-arctic-en-kr_lexicon.tsv",
                "l2-arctic",
                _KR_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/common-voice-accent-id_lexicon.tsv",
                "common-voice",
                _ID_L2,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/bookbot_en_v1-v2_lexicon.tsv",
                "bookbot",
                synthetic=True,
            ),
            _dictionary(
                "synthetic/bookbot_en_v1-v2_lexicon_zipformer.tsv",
                "bookbot",
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_WIKIPRON,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-us",
        display_name="English (United States)",
        base_language="en",
        territory="US",
        macroregion="north-america",
        dictionaries=(
            _EN_US_WIKIPRON_BROAD,
            _EN_US_WIKIPRON_NARROW,
            _dictionary(
                "cmudict-ipa/cmudict-0.7b-ipa-segmented.tsv",
                "cmudict",
                _US,
                "broad",
            ),
            _dictionary(
                "cmudict-ipa/librispeech-lexicon-200k-allothers-g2p-ipa.tsv",
                "librispeech",
                _US,
                "broad",
            ),
            _dictionary("synthetic/mfa_wu_us.tsv", "mfa-wu", _US, synthetic=True),
            _dictionary(
                "synthetic/mfa_wu_us_numbers.tsv",
                "mfa-wu",
                _US,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/common-voice-accent-us_lexicon.tsv",
                "common-voice",
                _US,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-us-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_US_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-us-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-us-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-uk",
        display_name="English (United Kingdom)",
        base_language="en",
        territory="GB",
        macroregion="europe",
        dictionaries=(
            _EN_UK_WIKIPRON_BROAD,
            _EN_UK_WIKIPRON_NARROW,
            _dictionary("synthetic/mfa_wu_uk.tsv", "mfa-wu", _UK, synthetic=True),
            _dictionary(
                "synthetic/mfa_wu_uk_numbers.tsv",
                "mfa-wu",
                _UK,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/common-voice-accent-gb_lexicon.tsv",
                "common-voice",
                _UK,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-uk-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_UK_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-uk-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-uk-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-au",
        display_name="English (Australia)",
        base_language="en",
        territory="AU",
        macroregion="oceania",
        dictionaries=(
            _EN_AU_WIKIPRON_BROAD,
            _EN_AU_WIKIPRON_NARROW,
            _dictionary("asr-data/austalk_en_au.tsv", "austalk", _AU),
            _dictionary("asr-data/sc_cw_en_au.tsv", "sc-cw", _AU),
            _dictionary("synthetic/mfa_wu_au.tsv", "mfa-wu", _AU, synthetic=True),
            _dictionary(
                "synthetic/mfa_wu_au_numbers.tsv",
                "mfa-wu",
                _AU,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/common-voice-accent-au_lexicon.tsv",
                "common-voice",
                _AU,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-au-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_AU_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-au-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-au-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-nz",
        display_name="English (New Zealand)",
        base_language="en",
        territory="NZ",
        macroregion="oceania",
        dictionaries=(
            _EN_NZ_WIKIPRON_BROAD,
            _EN_NZ_WIKIPRON_NARROW,
            _dictionary("synthetic/mfa_wu_nz.tsv", "mfa-wu", _NZ, synthetic=True),
            _dictionary(
                "synthetic/mfa_wu_nz_numbers.tsv",
                "mfa-wu",
                _NZ,
                synthetic=True,
            ),
            _dictionary(
                "synthetic/common-voice-accent-nz_lexicon.tsv",
                "common-voice",
                _NZ,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-nz-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_NZ_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-nz-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-nz-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-ca",
        display_name="English (Canada)",
        base_language="en",
        territory="CA",
        macroregion="north-america",
        dictionaries=(
            _EN_CA_WIKIPRON_BROAD,
            _EN_CA_WIKIPRON_NARROW,
            _dictionary(
                "synthetic/common-voice-accent-ca_lexicon.tsv",
                "common-voice",
                _CA,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-ca-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_CA_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-ca-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-ca-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
    LanguagePack(
        id="en-in",
        display_name="English (India)",
        base_language="en",
        territory="IN",
        macroregion="south-asia",
        dictionaries=(
            _EN_IN_WIKIPRON_BROAD,
            _EN_IN_WIKIPRON_NARROW,
            _dictionary(
                "synthetic/common-voice-accent-in_lexicon.tsv",
                "common-voice",
                _IN,
                synthetic=True,
            ),
        ),
        g2p_profiles=(
            G2pProfile(
                id="en-in-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_EN_IN_WIKIPRON_BROAD,
                model="bookbot/onnx-byt5-small-wikipron-eng-latn-in-broad-quantized-avx512_vnni",
            ),
        ),
        default_g2p_profile_id="en-in-wikipron-broad",
        text_normalizer=normalize_numbers,
        phoneme_normalizer=normalize_english_phonemes,
    ),
)

_SPANISH_PACKS = (
    LanguagePack(
        id="es",
        display_name="Spanish",
        base_language="es",
        territory=None,
        macroregion=None,
        dictionaries=(
            _ES_CHARSIU,
            _ES_WIKIPRON_CA_BROAD,
            _ES_WIKIPRON_CA_NARROW,
        ),
        g2p_profiles=(
            G2pProfile(
                id="es-charsiu-phonetic",
                backend="charsiu-g2p",
                transcription="phonetic",
                dictionary=_ES_CHARSIU,
                model=None,
            ),
            G2pProfile(
                id="es-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_ES_WIKIPRON_CA_BROAD,
                model=None,
            ),
            G2pProfile(
                id="es-wikipron-narrow",
                backend="wikipron",
                transcription="narrow",
                dictionary=_ES_WIKIPRON_CA_NARROW,
                model=None,
            ),
        ),
        default_g2p_profile_id="es-charsiu-phonetic",
        text_normalizer=normalize_spanish_text,
        phoneme_normalizer=None,
    ),
    LanguagePack(
        id="es-es",
        display_name="Spanish (Spain)",
        base_language="es",
        territory="ES",
        macroregion="europe",
        dictionaries=(
            _ES_CHARSIU,
            _ES_WIKIPRON_CA_BROAD,
            _ES_WIKIPRON_CA_NARROW,
        ),
        g2p_profiles=(
            G2pProfile(
                id="es-es-charsiu-phonetic",
                backend="charsiu-g2p",
                transcription="phonetic",
                dictionary=_ES_CHARSIU,
                model=None,
            ),
            G2pProfile(
                id="es-es-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_ES_WIKIPRON_CA_BROAD,
                model=None,
            ),
            G2pProfile(
                id="es-es-wikipron-narrow",
                backend="wikipron",
                transcription="narrow",
                dictionary=_ES_WIKIPRON_CA_NARROW,
                model=None,
            ),
        ),
        default_g2p_profile_id="es-es-charsiu-phonetic",
        text_normalizer=normalize_spanish_text,
        phoneme_normalizer=None,
    ),
    LanguagePack(
        id="es-419",
        display_name="Spanish (Latin America)",
        base_language="es",
        territory=None,
        macroregion="latin-america",
        dictionaries=(
            _ES_LATIN_CHARSIU,
            _ES_WIKIPRON_LA_BROAD,
            _ES_WIKIPRON_LA_NARROW,
        ),
        g2p_profiles=(
            G2pProfile(
                id="es-419-charsiu-phonetic",
                backend="charsiu-g2p",
                transcription="phonetic",
                dictionary=_ES_LATIN_CHARSIU,
                model=None,
            ),
            G2pProfile(
                id="es-419-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_ES_WIKIPRON_LA_BROAD,
                model=None,
            ),
            G2pProfile(
                id="es-419-wikipron-narrow",
                backend="wikipron",
                transcription="narrow",
                dictionary=_ES_WIKIPRON_LA_NARROW,
                model=None,
            ),
        ),
        default_g2p_profile_id="es-419-charsiu-phonetic",
        text_normalizer=normalize_spanish_text,
        phoneme_normalizer=None,
    ),
    LanguagePack(
        id="es-mx",
        display_name="Spanish (Mexico)",
        base_language="es",
        territory="MX",
        macroregion="latin-america",
        dictionaries=(_ES_MX_CHARSIU,),
        g2p_profiles=(
            G2pProfile(
                id="es-mx-charsiu-phonetic",
                backend="charsiu-g2p",
                transcription="phonetic",
                dictionary=_ES_MX_CHARSIU,
                model=None,
            ),
        ),
        default_g2p_profile_id="es-mx-charsiu-phonetic",
        text_normalizer=normalize_spanish_text,
        phoneme_normalizer=None,
    ),
    LanguagePack(
        id="es-co",
        display_name="Spanish (Colombia)",
        base_language="es",
        territory="CO",
        macroregion="latin-america",
        dictionaries=(
            _ES_LATIN_CHARSIU,
            _ES_WIKIPRON_LA_BROAD,
            _ES_WIKIPRON_LA_NARROW,
        ),
        g2p_profiles=(
            G2pProfile(
                id="es-co-charsiu-phonetic",
                backend="charsiu-g2p",
                transcription="phonetic",
                dictionary=_ES_LATIN_CHARSIU,
                model=None,
            ),
            G2pProfile(
                id="es-co-wikipron-broad",
                backend="wikipron",
                transcription="broad",
                dictionary=_ES_WIKIPRON_LA_BROAD,
                model=None,
            ),
            G2pProfile(
                id="es-co-wikipron-narrow",
                backend="wikipron",
                transcription="narrow",
                dictionary=_ES_WIKIPRON_LA_NARROW,
                model=None,
            ),
        ),
        default_g2p_profile_id="es-co-charsiu-phonetic",
        text_normalizer=normalize_spanish_text,
        phoneme_normalizer=None,
    ),
)


_LANGUAGE_PACKS = {pack.id: pack for pack in _ENGLISH_PACKS + _SPANISH_PACKS}


def supported_lexicon_languages() -> Tuple[str, ...]:
    return tuple(sorted(_LANGUAGE_PACKS))


def supported_g2p_languages(
    backend: Optional[str] = None,
    transcription: Optional[str] = None,
) -> Tuple[str, ...]:
    if backend is None and transcription is None:
        return tuple(
            sorted(
                pack.id
                for pack in _LANGUAGE_PACKS.values()
                if pack.text_normalizer is not None and pack.g2p_profiles
            )
        )

    selected_backend = backend or "wikipron"
    selected_transcription = transcription or "broad"
    return tuple(
        sorted(
            pack.id
            for pack in _LANGUAGE_PACKS.values()
            if pack.text_normalizer is not None
            and pack.g2p_profile(selected_backend, selected_transcription) is not None
        )
    )


def get_language_pack(lang: str) -> LanguagePack:
    try:
        return _LANGUAGE_PACKS[lang]
    except (KeyError, TypeError):
        choices = ", ".join(supported_lexicon_languages())
        raise ValueError(
            "Unsupported language {!r}. Supported languages: {}".format(lang, choices)
        ) from None
