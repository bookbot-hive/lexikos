from dataclasses import FrozenInstanceError

import pytest

from lexikos import G2p, Lexicon, OOVWarning, charsiu_prompt
from lexikos.languages import get_language_pack


@pytest.fixture(scope="module")
def en_us_lexicon():
    return Lexicon("en-us")


@pytest.fixture(scope="module")
def normalized_en_us_lexicon():
    return Lexicon("en-us", normalize_phonemes=True)


@pytest.fixture(scope="module")
def es_419_lexicon():
    return Lexicon("es-419")




def test_language_is_required_for_both_public_apis():
    with pytest.raises(TypeError):
        Lexicon()
    with pytest.raises(TypeError):
        G2p()


def test_language_ids_are_exact_and_errors_list_valid_choices():
    with pytest.raises(ValueError) as error:
        Lexicon("EN-US")

    message = str(error.value)
    assert "Unsupported language 'EN-US'" in message
    assert "en-us" in message
    assert "en-uk" in message


def test_supported_languages_are_discoverable():
    expected_lexicons = (
        "en",
        "en-au",
        "en-ca",
        "en-in",
        "en-nz",
        "en-uk",
        "en-us",
        "es",
        "es-419",
        "es-es",
        "es-mx",
    )
    expected_g2p = expected_lexicons
    expected_narrow_g2p = ("es", "es-419", "es-es")
    assert Lexicon.supported_languages() == expected_lexicons
    assert G2p.supported_languages() == expected_g2p
    assert G2p.supported_languages(narrow=True) == expected_narrow_g2p


def test_explicit_g2p_filters_select_exact_profiles():
    assert G2p.supported_languages(backend="wikipron", narrow=True) == (
        "es",
        "es-419",
        "es-es",
    )
    assert G2p("es-419").backend == "charsiu-g2p"
    assert G2p("es-419").transcription == "phonetic"
    assert G2p("es-419", narrow=False).transcription == "broad"
    assert G2p("es-419", backend="wikipron", narrow=True).transcription == "narrow"


def test_language_pack_carries_locale_metadata():
    generic = get_language_pack("en")
    assert generic.base_language == "en"
    assert generic.territory is None
    assert generic.macroregion is None

    united_states = get_language_pack("en-us")
    assert united_states.base_language == "en"
    assert united_states.territory == "US"
    assert united_states.macroregion == "north-america"



def test_spanish_text_normalization_preserves_numbers_and_accents():
    normalize = get_language_pack("es").text_normalizer
    assert normalize("corazo\u0301n 12") == "corazón 12"


def test_latin_american_spanish_lookup_has_source_metadata(es_419_lexicon):
    pronunciation = next(
        pronunciation
        for pronunciation in es_419_lexicon["corazón"]
        if pronunciation.ipa == "korason"
    )
    source = pronunciation.sources[0]
    assert source.source == "charsiu-g2p"
    assert source.language == "es-419"
    assert source.dialect.territory is None
    assert source.dialect.macroregion == "latin-america"
    assert source.transcription == "phonetic"
    assert not source.synthetic
    assert source.source_id
    assert source.observation_ids
    assert isinstance(source.observation_ids, tuple)
    assert source.source_revision
    assert source.evidence_status == "accepted"


def test_sqlite_lexicon_retains_mapping_behavior(es_419_lexicon):
    assert "corazón" in es_419_lexicon
    assert "zzzzlexikos" not in es_419_lexicon
    assert len(es_419_lexicon) > 0
    assert isinstance(next(iter(es_419_lexicon)), str)

    assert {item.ipa for item in es_419_lexicon["ayúdenme"]} == {
        "ajudɛn",
        "ajudɛnmɛ",
    }




def test_spanish_dictionary_only_g2p_warns_for_oov():
    g2p = G2p("es-419")
    assert g2p("corazón") == ["korason"]

    with pytest.warns(OOVWarning, match="no G2P model is available"):
        assert g2p("zzzzlexikos") == ["zzzzlexikos"]


def test_spanish_wikipron_broad_and_narrow_profiles_are_distinct():
    broad = G2p("es-419", backend="wikipron", narrow=False)
    narrow = G2p("es-419", backend="wikipron", narrow=True)

    assert broad("corazón") == ["k o ɾ a s o n"]
    assert narrow("corazón") == ["k o ɾ a s õ n"]


def test_charsiu_prompt_uses_locale_tag_normalization_and_required_spacing():
    assert charsiu_prompt("es", " AÑO ") == "<spa>: año"
    assert charsiu_prompt("es-419", "CORAZO\u0301N") == "<spa-latin>: corazón"
    assert charsiu_prompt("es-mx", "NIÑO") == "<spa-me>: niño"

    with pytest.raises(ValueError, match="Unsupported CharsiuG2P language 'en'"):
        charsiu_prompt("en", "hello")


def test_identical_ipa_has_all_sources(en_us_lexicon):
    pronunciation = next(
        pronunciation
        for pronunciation in en_us_lexicon["a"]
        if pronunciation.ipa == "ə"
    )

    assert {source.source for source in pronunciation.sources} == {
        "cmudict",
        "librispeech",
        "wikipron",
    }
    assert {source.language for source in pronunciation.sources} == {"en-us"}
    assert {source.dialect.group for source in pronunciation.sources} == {"american"}
    assert {source.transcription for source in pronunciation.sources} == {"broad"}
    assert not any(source.synthetic for source in pronunciation.sources)

    with pytest.raises(FrozenInstanceError):
        setattr(pronunciation.sources[0], "source", "changed")


def test_normalization_unions_sources_when_ipa_collapses(
    normalized_en_us_lexicon,
):
    pronunciation = next(
        pronunciation
        for pronunciation in normalized_en_us_lexicon["aachen"]
        if pronunciation.ipa == "ɑ k ə n"
    )

    assert {source.source for source in pronunciation.sources} == {
        "cmudict",
        "librispeech",
        "wikipron",
    }


def test_synthetic_sources_remain_opt_in(en_us_lexicon):
    assert not any(
        source.synthetic
        for pronunciation in en_us_lexicon["hello"]
        for source in pronunciation.sources
    )

    lexicon_with_synthetic = Lexicon("en-us", include_synthetic=True)
    assert any(
        source.synthetic
        for pronunciation in lexicon_with_synthetic["hello"]
        for source in pronunciation.sources
    )


def test_g2p_selects_a_complete_broad_language_pack():
    assert G2p("en-us")("hello") == ["h ɛ l o ʊ"]


def test_g2p_rejects_a_language_without_requested_model_width():
    with pytest.raises(ValueError, match="does not support narrow G2P"):
        G2p("en-us", narrow=True)


def test_g2p_uses_and_reuses_model_for_unknown_words(monkeypatch):
    g2p = G2p("en-us")
    model_loads = []

    def load_model():
        model_loads.append(True)
        return lambda token: "model:" + token

    monkeypatch.setattr(g2p, "_get_t5_model", load_model)

    assert g2p("zzzzlexikos zzzzlexikos") == [
        "model:zzzzlexikos",
        "model:zzzzlexikos",
    ]
    assert model_loads == [True]
