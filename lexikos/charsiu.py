"""CharsiuG2P language tags and multilingual prompt construction."""

from typing import Dict

from .languages import get_language_pack


CHARSIU_LANGUAGE_TAGS: Dict[str, str] = {
    "es": "spa",
    "es-es": "spa",
    "es-419": "spa-latin",
    "es-mx": "spa-me",
    "es-co": "spa-co",
}


def charsiu_prompt(lang: str, text: str) -> str:
    """Build the normalized ``<language>: text`` input used by CharsiuG2P."""
    try:
        tag = CHARSIU_LANGUAGE_TAGS[lang]
    except (KeyError, TypeError):
        choices = ", ".join(CHARSIU_LANGUAGE_TAGS)
        raise ValueError(
            "Unsupported CharsiuG2P language {!r}. Supported languages: {}".format(
                lang, choices
            )
        ) from None

    pack = get_language_pack(lang)
    normalized = pack.text_normalizer(text).strip().casefold()
    return "<{}>: {}".format(tag, normalized)
