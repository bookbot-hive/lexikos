# Lexikos

Lexikos is a provenance-preserving pronunciation lexicon and
grapheme-to-phoneme (G2P) package. It serves immutable pronunciation records
from a deterministic, read-only SQLite runtime and keeps every IPA value linked
to its source, revision, dialect, transcription type, and observation IDs.

<p align="center">
  <img src="https://github.com/bookbot-hive/lexikos/raw/main/assets/lexikos.png" alt="Lexikos logo" width="300"/>
</p>

## Current release boundary

The current data edition is **`2026.09.4`**. Code and data artifacts have
different distribution boundaries:

- Source code, configuration, tests, and the release manifest are tracked in
  Git.
- `lexikos/data/runtime.sqlite3` is generated, ignored by Git, and bundled into
  release wheels.
- The append-only curation database and raw source TSVs remain external.
- The verified wheel is approximately 950 MB, so it is not suitable for
  ordinary PyPI distribution.

The data-backed `2026.09.4` wheel is published as an asset of the
[`v1.0.0` GitHub release](https://github.com/bookbot-hive/lexikos/releases/tag/v1.0.0).
It is not published on PyPI; download and verify the release asset explicitly.

### Installing a release wheel

When a matching wheel is published as a GitHub Release asset, download the
wheel and its checksum from the same release, verify it, and install the local
file:

```sh
sha256sum lexikos-1.0.0-py3-none-any.whl
python -m pip install ./lexikos-1.0.0-py3-none-any.whl
```

Use the checksum in the release notes or accompanying checksum file. The
README is embedded into wheel metadata, so it cannot be the authoritative
record of the wheel's own hash.

### Development checkout

```sh
git clone https://github.com/bookbot-hive/lexikos.git
cd lexikos
python -m pip install -e .
```

An editable checkout does not contain `runtime.sqlite3`. Runtime-backed
`Lexicon` and `G2p` calls require either a generated snapshot at
`lexikos/data/runtime.sqlite3` or an installed release wheel containing it.
See [Rebuilding the data edition](#rebuilding-the-data-edition).

## Explicit language selection

The multilingual revamp made language selection mandatory:

```py
Lexicon(language_id)
G2p(language_id)
```

Calling `Lexicon()` or `G2p()` without a language ID raises `TypeError`.
Language IDs are exact lowercase BCP-47-style identifiers such as `en-us`,
`es-419`, and `es-mx`; aliases such as `EN-US` or `en_US` are not accepted.

```py
>>> from lexikos import G2p, Lexicon
>>> Lexicon.supported_languages()
('en', 'en-au', 'en-ca', 'en-in', 'en-nz', 'en-uk', 'en-us', 'es', 'es-419', 'es-es', 'es-mx')
>>> G2p.supported_languages()
('en', 'en-au', 'en-ca', 'en-in', 'en-nz', 'en-uk', 'en-us', 'es', 'es-419', 'es-es', 'es-mx')

```


## Lexicon API

`Lexicon` is a language-specific, read-only mapping:

```py
Lexicon(
    language_id,
    normalize_phonemes=False,
    include_synthetic=False,
)
```

`lexicon[word]` returns one immutable `Pronunciation` for each distinct IPA
value. Each pronunciation has an immutable `sources` tuple. When multiple
observations provide the same IPA, Lexikos returns the IPA once and preserves
all contributing sources instead of selecting one winner.

```py
>>> from lexikos import Lexicon
>>> pronunciations = Lexicon("es-419")["corazón"]
>>> for item in pronunciations:
...     print(
...         item.ipa,
...         [(source.source, source.transcription) for source in item.sources],
...     )
k o ɾ a s o n [('wikipron', 'broad')]
k o ɾ a s õ n [('wikipron', 'narrow')]
korason [('charsiu-g2p', 'phonetic')]

```

Every `PronunciationSource` also carries:

- the exact Lexikos language pack;
- structured territory, macroregion, group, locality, and dialect features;
- stable source and observation IDs;
- source revision and source URL when available;
- license references and evidence status;
- whether the evidence is synthetic.

Synthetic evidence is excluded by default:

```py
lexicon = Lexicon("en-us", include_synthetic=True)
```

Phoneme normalization is opt-in and available only for packs that declare a
phoneme normalizer:

```py
lexicon = Lexicon("en-us", normalize_phonemes=True)
```

## G2P API

`G2p` also requires an explicit language ID:

```py
G2p(
    language_id,
    backend=None,
    narrow=None,
    normalize_phonemes=False,
)
```

With no filters, `G2p(language_id)` selects that pack's declared default
profile. English defaults can use neural fallback. Current Spanish profiles are
dictionary-only: an out-of-vocabulary token emits `OOVWarning` and passes
through unchanged rather than inventing a pronunciation.

```py
>>> from lexikos import G2p
>>> G2p("es-419")("corazón")
['korason']
>>> G2p("es-419", backend="wikipron", narrow=False)("corazón")
['k o ɾ a s o n']
>>> G2p("es-419", backend="wikipron", narrow=True)("corazón")
['k o ɾ a s õ n']

```

Backend and width filters select an exact configured profile; they do not
silently fall back to another source.

## Spanish packs

| Language ID | Lexicon evidence | G2P profiles | Default |
| --- | --- | --- | --- |
| `es` | Charsiu `spa`; WikiPron Castilian broad/narrow | Charsiu phonetic; WikiPron broad/narrow | Charsiu phonetic |
| `es-es` | Charsiu `spa`; WikiPron Castilian broad/narrow | Charsiu phonetic; WikiPron broad/narrow | Charsiu phonetic |
| `es-419` | Charsiu `spa-latin`; WikiPron Latin-American broad/narrow | Charsiu phonetic; WikiPron broad/narrow | Charsiu phonetic |
| `es-mx` | Charsiu `spa-me` | Charsiu phonetic | Charsiu phonetic |

Important boundaries:

- Generic `es` is backed by peninsular evidence; it is not dialect-neutral.
- `es-419` preserves the union of pooled Latin-American Charsiu and WikiPron
  evidence.
- `es-mx` uses the Mexican-specific Charsiu source.

Charsiu's pinned language registry contains `spa`, `spa-latin`, and `spa-me`.
The prompt helper uses the exact locale-to-model mapping:

```py
>>> from lexikos import charsiu_prompt
>>> charsiu_prompt("es-419", "CORAZÓN")
'<spa-latin>: corazón'

```

`charsiu_prompt` only constructs input for external Charsiu training or model
serving. Runtime dictionary-backed `G2p` does not call it.

## Storage and provenance architecture

Lexikos uses two SQLite artifacts.

### Curation database

The external curation database is append-only. It records:

- source and import-run identities;
- exact raw payload URIs, revisions, and SHA-256 hashes;
- immutable raw observations;
- accepted, rejected, and superseded reviews;
- canonical pronunciation-to-evidence links.

Rejected but capturable source rows remain available for audit. Re-importing
the same pinned source is idempotent.

### Runtime database

The wheel contains a deterministic runtime snapshot with:

- unique pronunciations grouped by language pack, word, and IPA;
- immutable grouped evidence for every pronunciation;
- exact G2P profiles and dictionary rows;
- build, configuration, parser, source, and count metadata.

Runtime connections use SQLite read-only mode and `PRAGMA query_only = ON`.
Source TSVs are not packaged in the wheel.

### Edition `2026.09.4`

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| Curation SQLite | `68dcc8e6442419a38a2e8d8ec9c0027d640f1b4fc06d2c52d908932157d7a5da` | 5,632,610,304 bytes |
| Runtime SQLite | `e89ea17a904b856a00cdf3705c46ce3324280282780cb2277dcfe65ae357655e` | 3,106,291,712 bytes |

The tracked
[`release-manifest.json`](./lexikos/data/release-manifest.json) is the
machine-readable source of truth for source revisions, payload hashes, licenses,
build environment, configuration revision, and runtime identity.

## Rebuilding the data edition

Raw dictionaries and the curation database are external build inputs. The
source root must contain every path declared by `lexikos/languages.py` and
`release-manifest.json`.

```sh
python scripts/build_lexicon_databases.py \
    --source-root /path/to/source-snapshot \
    --curation-db /path/to/curation.sqlite3 \
    --runtime-db lexikos/data/runtime.sqlite3 \
    --release-manifest lexikos/data/release-manifest.json
```

The compiler:

1. verifies the pinned Python and SQLite build environment;
2. verifies every raw payload hash and active import run;
3. preserves malformed-but-capturable rows as rejected observations;
4. compiles deterministic pronunciation, evidence, and G2P tables;
5. runs foreign-key and SQLite integrity checks;
6. refuses publication when the rebuilt runtime hash differs from the
   manifest.

The current manifest pins Python `3.13.2` and SQLite `3.45.3`.

## Preparing Charsiu training data

The preparation CLI converts external `word<TAB>IPA` dictionaries into
deterministic, word-disjoint Charsiu train/dev/test files:

```sh
python scripts/prepare_charsiu_g2p.py \
    /path/to/source-snapshot/charsiu/spa-latin.tsv \
    --language es-419 \
    --output-dir prepared/es-419
```

It writes:

```text
prepared/es-419/train/spa-latin.tsv
prepared/es-419/dev/spa-latin.tsv
prepared/es-419/test/spa-latin.tsv
prepared/es-419/manifest.json
```

The manifest records source and output hashes. The language choice is
constrained to locales backed by the pinned Charsiu model contract.

The pinned upstream trainer omits the required space after its language prefix.
Before training, change both prefix expressions in CharsiuG2P
`src/data_utils.py` from:

```py
'<' + language + '>:' + word
```

to:

```py
'<' + language + '>: ' + word
```

Without this correction, training and `charsiu_prompt` use different input
formats.

## Data rights

The repository Apache-2.0 license covers Lexikos software; it does not
relicense third-party pronunciation data.

- WikiPron data is derived from Wiktionary and retains applicable attribution
  and share-alike obligations.
- Charsiu's pinned per-file records identify `spa` and `spa-me` as MIT
  ipa-dict derivatives and `spa-latin` as an Apache-2.0 Santiago Spanish
  Lexicon derivative.
- Other English sources retain their own terms and synthetic-status markers.

See [`lexikos/data/NOTICE.txt`](./lexikos/data/NOTICE.txt) and the release
manifest for exact source revisions and rights records.

## Development

Run the regression suite:

```sh
python -m pytest
```

Check the maintained Python surface:

```sh
python -m ruff check \
    lexikos/languages.py \
    lexikos/charsiu.py \
    lexikos/lexicon.py \
    lexikos/g2p.py \
    lexikos/storage.py \
    scripts/build_lexicon_databases.py \
    scripts/prepare_charsiu_g2p.py \
    tests
```

## Resources

- [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- [WikiPron](https://github.com/CUNY-CL/wikipron)
- [CMU Pronouncing Dictionary IPA](https://github.com/menelik3/cmudict-ipa)
- [OpenSLR SLR34](https://www.openslr.org/34/)

## License

Lexikos software is licensed under Apache-2.0. Bundled pronunciation data
retains the source-specific terms documented in `NOTICE.txt` and the release
manifest.