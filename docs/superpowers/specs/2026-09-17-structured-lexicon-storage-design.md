# Structured Lexicon Storage Design

**Date:** 2026-09-17  
**Status:** Approved; implementation pending
**Scope:** Lexicon ingestion, curation, provenance, runtime snapshots, and migration from TSV

## Problem

Lexikos currently stores pronunciation rows as `word<TAB>IPA` TSV files. Dataset-level metadata lives separately in `lexikos/languages.py`. This cannot faithfully represent metadata that varies by scraped occurrence, including source URL, upstream revision, raw language or dialect labels, license, parser version, review decision, corrections, or multiple observations supporting the same pronunciation.

The current runtime reconstructs provenance by applying one `DictionarySource` record to every row in a file. That is sufficient for uniform legacy files but not for heterogeneous WikiPron or other scraped sources. It also loads whole dictionaries into memory instead of querying an indexed artifact.

## Goals

1. Preserve every imported pronunciation observation without overwriting raw evidence.
2. Store exact source labels alongside normalized language, dialect, locality, and phonological metadata.
3. Retain parser, mapping, review, correction, licensing, and source-revision provenance.
4. Deduplicate canonical pronunciations without discarding their many-to-many evidence.
5. Keep locale-pack assignment separate from source dialect evidence.
6. Preserve the current public lookup contract: `Lexicon(lang)[word]` returns immutable `Pronunciation` records.
7. Generate a deterministic, indexed, read-only runtime SQLite database for the Python wheel.
8. Publish reproducible data editions without committing frequently changing SQLite binaries to ordinary Git history.

## Non-goals

- Implementing new WikiPron or website scrapers in this change.
- Training or deploying Spanish neural G2P models.
- Inventing missing dialect, accent, locality, feature, or license information.
- Exposing rejected observations or raw scrape payloads through the main runtime API.
- Adding a hosted curation service, PostgreSQL, Dolt, DVC, Git LFS, or a data lake.
- Supporting both TSV and SQLite runtime loaders indefinitely.

## Decisions

- Canonical curation store: normalized SQLite database.
- Runtime store: separately generated read-only SQLite database.
- Curation schema: six core tables.
- Evidence retention: every raw observation is immutable; interpretations and decisions are append-only reviews.
- Dialect metadata: preserve raw source labels and normalized mappings with field-level mapping provenance.
- Public API: retain `Pronunciation` results and enrich `PronunciationSource`.
- Locale packs: Git-reviewed configuration, never inferred from a requested frontend ID.
- Data publication: hash-named GitHub Release assets plus Git-tracked manifests.
- Raw ingestion exchange: append-friendly JSONL records and content-addressed raw payloads.

## Architecture

```text
Scraper/importer
  ├── observations-<sha256>.jsonl.zst
  └── raw-payloads-<sha256>.tar.zst
              │
              ▼
Canonical curation SQLite
  ├── immutable observations
  ├── append-only reviews and corrections
  ├── canonical pronunciations
  └── many-to-many evidence
              │
              │ deterministic snapshot compiler
              ▼
Runtime SQLite
  ├── accepted evidence only
  ├── locale-pack assignments
  ├── indexed normalized lookup
  └── snapshot provenance
              │
              ▼
Python wheel and GitHub data release
```

The curation database is not shipped to application users. The runtime database contains only data necessary for lookup and accepted attribution.

## Raw ingestion records

A scraper or importer emits one JSONL row per extracted pronunciation variant
from each physical source row before database import. The transport schema
includes:

```json
{
  "source_id": "wikipron",
  "source_revision": "upstream-revision-or-snapshot-id",
  "source_url": "https://example.test/entry",
  "source_row_reference": "page-or-row-identifier",
  "source_occurrence": 42,
  "variant_occurrence": 0,
  "word_raw": "niño",
  "pronunciation_raw": "/ˈniɲo/",
  "pronunciation_variant_raw": "/ˈniɲo/",
  "metadata_raw": {
    "language": "Spanish",
    "region": "Colombia",
    "accent": null,
    "options": {}
  },
  "retrieved_at": "2026-09-17T10:00:00Z",
  "raw_payload_sha256": "..."
}
```

Raw HTML, API responses, or source files are stored outside SQLite by SHA-256. JSONL refers to those payloads by hash. Import must not mutate either artifact.

## Canonical curation schema

SQLite foreign keys are enabled for every connection with `PRAGMA foreign_keys = ON`.

### 1. `source`

Stable attribution identity.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `name` | TEXT | NOT NULL |
| `homepage_url` | TEXT | NULL allowed |

`id` is a stable lowercase identifier such as `wikipron` or `charsiu-g2p`. Mutable upstream facts are captured on `import_run`, not rewritten into historical observations.

### 2. `import_run`

One source snapshot processed with one parser version.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `source_id` | TEXT | NOT NULL REFERENCES `source(id)` |
| `source_revision` | TEXT | NOT NULL |
| `license_id` | TEXT | NULL allowed |
| `license_url` | TEXT | NULL allowed |
| `redistribution_policy` | TEXT | NOT NULL |
| `parser_name` | TEXT | NOT NULL |
| `parser_version` | TEXT | NOT NULL |
| `raw_payload_uri` | TEXT | NOT NULL |
| `raw_payload_sha256` | TEXT | NOT NULL |
| `started_at` | TEXT | NOT NULL ISO-8601 UTC |
| `completed_at` | TEXT | NOT NULL ISO-8601 UTC |

`redistribution_policy` is one of `public`, `private`, `metadata-only`, or `unknown`. Unknown licensing permits local curation but blocks publication of the affected payload and pronunciation evidence until reviewed.

### 3. `observation`

Immutable source evidence.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `import_run_id` | TEXT | NOT NULL REFERENCES `import_run(id)` |
| `source_url` | TEXT | NULL allowed |
| `source_row_reference` | TEXT | NOT NULL |
| `source_occurrence` | INTEGER | NOT NULL, unique physical-row ordinal within one import run |
| `variant_occurrence` | INTEGER | NOT NULL, zero-based variant ordinal within the source row |
| `word_raw` | TEXT | NOT NULL |
| `pronunciation_raw` | TEXT | NOT NULL |
| `pronunciation_variant_raw` | TEXT | NOT NULL |
| `metadata_raw_json` | TEXT | NOT NULL, valid JSON object |
| `retrieved_at` | TEXT | NOT NULL ISO-8601 UTC |

The observation ID is SHA-256 over the import-run ID, source row reference,
source occurrence, variant occurrence, and canonical bytes of the raw record.
Importers assign a stable physical-row ordinal or byte offset to
`source_occurrence` and preserve extracted variant order in
`variant_occurrence`. Duplicate identical physical rows therefore remain
distinct, as do multiple pronunciations in one source row.

Unique constraint: `(import_run_id, source_occurrence, variant_occurrence)`.

`pronunciation_raw` preserves the complete source field unchanged.
`pronunciation_variant_raw` preserves the exact trimmed variant selected by
`variant_occurrence`. A review interprets that selected variant. The importer
must be able to reproduce the variant from the complete field under its
recorded parser version.

Observation rows are never updated or deleted. Re-importing the same immutable row is idempotent.

### 4. `review`

Append-only interpretation, mapping, and decision history.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `observation_id` | TEXT | NOT NULL REFERENCES `observation(id)` |
| `parser_version` | TEXT | NOT NULL |
| `language_normalized` | TEXT | NULL allowed |
| `word_normalized` | TEXT | NULL allowed |
| `ipa_normalized` | TEXT | NULL allowed |
| `metadata_normalized_json` | TEXT | NOT NULL, valid JSON object |
| `mapping_version` | TEXT | NOT NULL |
| `decision` | TEXT | NOT NULL: `accepted`, `rejected`, or `superseded` |
| `reason` | TEXT | NULL allowed |
| `reviewer` | TEXT | NOT NULL |
| `created_at` | TEXT | NOT NULL ISO-8601 UTC |
| `supersedes_review_id` | TEXT | UNIQUE, NULL or REFERENCES `review(id)` |

A correction inserts a new review referencing the prior review. No review is mutated. An active review is the terminal review in a correction chain. Only terminal reviews with decision `accepted` can contribute to a runtime snapshot.

Normalized metadata is a JSON object validated against a versioned JSON Schema. Every mapped field carries its raw value, normalized value, rule, confidence, and status:

```json
{
  "territory": {
    "raw": "Colombia",
    "value": "CO",
    "rule": "country-name-v1",
    "confidence": 1.0,
    "status": "mapped"
  },
  "macroregion": {
    "raw": null,
    "value": "latin-america",
    "rule": "territory-to-macroregion-v1",
    "confidence": 1.0,
    "status": "mapped"
  },
  "dialect_group": {
    "raw": null,
    "value": null,
    "rule": null,
    "confidence": null,
    "status": "unknown"
  },
  "locality": {
    "raw": null,
    "value": null,
    "rule": null,
    "confidence": null,
    "status": "unknown"
  },
  "features": []
}
```

Allowed status values are `mapped`, `ambiguous`, `unmapped`, and `unknown`. Country or pack IDs never imply dialect groups or phonological features.

### 5. `pronunciation`

Canonical accepted pronunciation identity.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `language` | TEXT | NOT NULL |
| `normalized_word` | TEXT | NOT NULL |
| `normalized_ipa` | TEXT | NOT NULL |

Unique constraint: `(language, normalized_word, normalized_ipa)`.

The ID is deterministically derived from that unique tuple. Creating an accepted review transactionally creates or reuses the canonical pronunciation.

### 6. `pronunciation_evidence`

Many-to-many provenance between canonical pronunciations and accepted reviews.

| Column | Type | Constraint |
| --- | --- | --- |
| `pronunciation_id` | TEXT | NOT NULL REFERENCES `pronunciation(id)` |
| `accepted_review_id` | TEXT | NOT NULL REFERENCES `review(id)` |

Primary key: `(pronunciation_id, accepted_review_id)`.

Historical links remain in curation. The snapshot compiler includes only evidence whose review is currently terminal and accepted. If a correction changes the normalized word or IPA, the new review links to the new canonical pronunciation; the old pronunciation disappears from runtime when it has no active evidence.

## Curation invariants and merge semantics

1. Raw payloads, JSONL rows, observations, and reviews are append-only.
2. Normalization never changes `word_raw`, `pronunciation_raw`,
   `pronunciation_variant_raw`, or `metadata_raw_json`.
3. Canonical identity is exact normalized language, word, and IPA.
4. Multiple source occurrences of the same canonical pronunciation remain separate evidence.
5. Conflicting dialect claims remain separate evidence; the compiler does not silently resolve them.
6. Missing values remain unknown. No locale, dialect, accent, transcription classification, license, or feature is invented.
7. Pack membership is not evidence and cannot rewrite evidence metadata.
8. All writes that accept a review and link evidence occur in one transaction.
9. `PRAGMA foreign_key_check` and `PRAGMA integrity_check` must pass before snapshot compilation.

## Locale-pack configuration

Locale packs remain reviewable text configuration in Git, for example:

```text
lexikos/packs/en-us.yaml
lexikos/packs/es-es.yaml
lexikos/packs/es-419.yaml
lexikos/packs/es-mx.yaml
lexikos/packs/es-co.yaml
```

A pack file declares:

- Pack ID and display metadata.
- Normalization policy.
- Included source/import filters.
- Accepted transcription classifications.
- Explicit evidence-selection rules.
- Synthetic-data inclusion policy.
- Default G2P profile ID. Every pack with at least one runtime dictionary and a
  text normalizer must name exactly one default profile.
- One or more G2P profiles for such packs. Each profile declares a stable ID,
  backend, transcription, optional model, exact dictionary source/import run,
  accepted-review filter, extraction-rule version, row/variant ordering,
  synthetic-evidence policy, duplicate-word policy, and lookup-selection
  policy.

For example:

```yaml
default_g2p_profile_id: en-us-wikipron-broad
g2p_profiles:
  - id: en-us-wikipron-broad
    backend: wikipron
    transcription: broad
    model: bookbot/onnx-byt5-small-wikipron-eng-latn-us-broad-quantized-avx512_vnni
    dictionary:
      source_id: wikipron
      import_run_id: wikipron-eng-latn-us-broad-2026-09-17
      review_filter: terminal-accepted
      include_synthetic: false
      extraction_rule: legacy-g2p-v1
      order: [source_occurrence, variant_occurrence]
      duplicate_word_policy: append
      lookup_selection: last
```

The named import run must appear in the release manifest and belong to the
declared source. Pack validation rejects a dictionary-plus-normalizer pack
without exactly one valid default profile; dictionary choice is never inferred
when several sources exist. A profile's `model` may be `null`. Such a
dictionary-only default remains supported and advertised; absence of a neural
fallback does not exclude the pack.

`legacy-g2p-v1` reproduces the current loader exactly:

1. Lowercase the raw word without other word normalization.
2. For each variant observation, split its complete `pronunciation_raw` field
   first on exact `" ~ "`, then on every comma, trim each variant, and reject
   empty variants.
3. Validate that `pronunciation_variant_raw` equals the split variant at that
   observation's `variant_occurrence`.
4. Replace exact `" . "` substrings in the selected variant with one space;
   perform no other period or whitespace normalization.
5. Emit only that observation's selected variant. The compiler consumes each
   variant observation once and never re-emits the complete split list for
   every sibling observation.
6. Preserve physical-row order and within-row variant order.
7. Append variants from repeated words and select the last appended value on
   lookup.

Other extraction rules require their own versioned, tested contract. The
compiler rejects profiles with an unpinned import run, source mismatch,
unsupported extraction rule, or ordering/selection policy it cannot execute.

Pack configuration selects accepted evidence; it never changes its dialect metadata. The reserved `es-co` pack accepts only explicitly Colombian evidence and remains empty until such a source is reviewed; pooled Latin-American evidence belongs to `es-419`.

## Runtime SQLite schema

The package ships one database containing all locale packs to avoid duplicating shared evidence.

Every runtime connection enables `PRAGMA foreign_keys = ON`. The compiler runs
`PRAGMA foreign_key_check` and `PRAGMA integrity_check` before publication.

### `snapshot_metadata`

Key/value records for:

- Snapshot format version.
- Curation schema version.
- Included import-run IDs and payload hashes.
- Parser and mapping versions.
- Locale-pack configuration Git commit.
- Snapshot compiler version.
- Python and SQLite versions.
- Build timestamp supplied by the release manifest.
- Database SHA-256 recorded externally in the manifest.

### `pronunciation`

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | INTEGER | PRIMARY KEY |
| `pack_id` | TEXT | NOT NULL |
| `language` | TEXT | NOT NULL |
| `normalized_word` | TEXT | NOT NULL |
| `ipa` | TEXT | NOT NULL |
| `phoneme_normalized_ipa` | TEXT | NULL allowed |

Unique constraint: `(pack_id, normalized_word, ipa)`. Lookup index:
`(pack_id, normalized_word)`. `ipa` is the representation returned when
`normalize_phonemes=False`. When the pack has a phoneme normalizer,
`phoneme_normalized_ipa` stores its output. A normalized lookup groups rows by
that output and unions their evidence, preserving the current collapse of
distinct base pronunciations. Requesting normalization for a pack without a
normalizer continues to raise `ValueError`.

### `evidence`

One row per unique accepted attribution record for a runtime pronunciation:

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `pronunciation_id` | INTEGER | NOT NULL REFERENCES `pronunciation(id)` |
| `source_id` | TEXT | NOT NULL |
| `source_name` | TEXT | NOT NULL |
| `language` | TEXT | NOT NULL |
| `source_language` | TEXT | NULL allowed |
| `source_language_raw` | TEXT | NULL allowed |
| `observation_ids_json` | TEXT | NOT NULL, valid non-empty JSON array |
| `source_url` | TEXT | NULL allowed |
| `source_revision` | TEXT | NOT NULL |
| `license_id` | TEXT | NULL allowed |
| `license_url` | TEXT | NULL allowed |
| `evidence_status` | TEXT | NOT NULL |
| `dialect_json` | TEXT | NULL or valid JSON object |
| `transcription` | TEXT | NOT NULL |
| `synthetic` | INTEGER | NOT NULL, CHECK value is 0 or 1 |

`id` is a deterministic SHA-256 over the pronunciation identity, canonical
attribution metadata, and sorted observation IDs. This prevents duplicate
grouped evidence even when nullable attribution fields are present.

Rows with identical accepted attribution metadata are grouped, while
`observation_ids_json` retains every contributing occurrence ID in sorted
order.

`source_name` supplies the existing human-readable `PronunciationSource.source`
field; `source_id` is the stable machine identifier. `language` is the runtime
pack context and preserves the existing `PronunciationSource.language`
contract. `source_language` and `source_language_raw` retain, respectively,
the normalized and exact source-evidence language. One accepted observation
may therefore appear under multiple pack-context `language` values without
changing its source-language evidence.

### `g2p_profile`

Model-profile metadata compiled from Git-reviewed pack configuration:

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | TEXT | PRIMARY KEY |
| `pack_id` | TEXT | NOT NULL |
| `backend` | TEXT | NOT NULL |
| `transcription` | TEXT | NOT NULL |
| `model` | TEXT | NULL allowed |
| `is_default` | INTEGER | NOT NULL, CHECK value is 0 or 1 |
| `dictionary_source_id` | TEXT | NOT NULL |
| `dictionary_import_run_id` | TEXT | NOT NULL |
| `extraction_rule` | TEXT | NOT NULL |
| `config_path` | TEXT | NOT NULL |
| `config_sha256` | TEXT | NOT NULL |
| `dictionary_policy_json` | TEXT | NOT NULL, canonical JSON |

Unique constraint: `(pack_id, backend, transcription)`. A partial unique index
on `pack_id WHERE is_default = 1` permits exactly one default row per pack; the
compiler separately rejects a required pack with no default row.

### `g2p_dictionary`

The exact dictionary view used before neural fallback:

| Column | Type | Constraint |
| --- | --- | --- |
| `profile_id` | TEXT | NOT NULL REFERENCES `g2p_profile(id)` |
| `lookup_word` | TEXT | NOT NULL |
| `ordinal` | INTEGER | NOT NULL |
| `output_ipa` | TEXT | NOT NULL |
| `observation_id` | TEXT | NOT NULL |
| `accepted_review_id` | TEXT | NOT NULL |

Primary key: `(profile_id, lookup_word, ordinal)`. A lookup index covers
`(profile_id, lookup_word)`. Rows are derived only from terminal accepted
reviews in the profile's declared import run. `ordinal` is assigned from the
declared row/variant ordering. `observation_id` and `accepted_review_id`
preserve the exact curation evidence for each output. The highest matching
ordinal is selected because the declared compatibility policy is `last`.

## Runtime API

The public lookup shape remains:

```python
lexicon = Lexicon("es-mx")
pronunciations = lexicon["niño"]
```

`Pronunciation` remains an immutable IPA plus sources.
`PronunciationSource` gains these typed fields:

```python
source_id: str
observation_ids: tuple[str, ...]
source_url: str | None
license_id: str | None
license_url: str | None
source_revision: str
evidence_status: str
source_language: str | None
source_language_raw: str | None
```

`observation_ids` is constructed from the sorted runtime JSON array as an
immutable tuple. Unknown source languages, source URLs, and license values
remain `None`; empty strings are not substitutes for unknown values.

Existing fields remain:

```python
source: str
language: str
dialect: Dialect | None
transcription: str
synthetic: bool
```

`Lexicon` becomes a read-only mapping backed by indexed SQLite queries rather
than loading every row into a `UserDict`. `__getitem__`, `__contains__`,
`__iter__`, and `__len__` preserve mapping behavior. Missing words continue to
raise `KeyError`.

With `include_synthetic=False`, the evidence query excludes synthetic rows
before IPA grouping and omits any pronunciation left without evidence. With
`include_synthetic=True`, synthetic and non-synthetic evidence participate,
then sources are deduplicated and sorted normally. This filtering precedes
both base-IPA and phoneme-normalized grouping.

The runtime opens the packaged database read-only. It exposes accepted provenance only. Curation history requires the separate curation artifact and tooling.

`G2p` reads its profile, ordered dictionary outputs, and optional model
identifier from `g2p_profile` and `g2p_dictionary` in the same runtime
snapshot. Its normalized token queries `lookup_word`; a known-word hit returns
the `output_ipa` at the highest ordinal. When a model is configured, an unknown
word uses that model as before.

For a dictionary-only profile, an unknown word emits
`OOVWarning(UserWarning)` with the language, token, and “no G2P model
available” reason, then returns the normalized token unchanged in the
`List[str]` result. The pass-through token is not processed by the phoneme
normalizer. This preserves the return shape without inventing a pronunciation.
A configured model that fails to load remains an error; it is not treated as
an intentionally model-less profile.

The selection API uses `None` to distinguish default-profile resolution from
the existing explicit filters:

```python
G2p(
    lang: str,
    *,
    backend: str | None = None,
    narrow: bool | None = None,
    normalize_phonemes: bool = False,
)

G2p.supported_languages(
    backend: str | None = None,
    narrow: bool | None = None,
) -> tuple[str, ...]
```

The combinations are:

| `backend` | `narrow` | Selection |
| --- | --- | --- |
| `None` | `None` | Pack default profile |
| non-`None` | `None` | That backend's broad profile |
| non-`None` | `False` | That backend's broad profile |
| non-`None` | `True` | That backend's narrow profile |
| `None` | `False` | WikiPron broad, preserving the explicit legacy call |
| `None` | `True` | WikiPron narrow, preserving the explicit legacy call |

`G2p(lang)` therefore selects the declared default even when it is
non-WikiPron or non-broad. Explicit selectors require an exact matching
profile and raise `ValueError` when none exists.
`G2p.supported_languages()` applies the same table: the no-argument call lists
packs with defaults, while any supplied filter lists exact matches. No G2P
code reads packaged TSV files after cutover.

## Deterministic snapshot generation

The compiler:

1. Reads a Git-tracked release manifest naming exact import-run IDs.
2. Verifies all raw-payload hashes and database integrity.
3. Resolves terminal accepted reviews.
4. Applies the locale-pack configuration at the recorded Git commit.
5. Groups canonical pronunciations and attribution-equivalent evidence.
6. Compiles each G2P profile from its exact source/import-run declaration and
   versioned extraction rule, retaining observation provenance and order.
7. Creates a new runtime database from scratch.
8. Uses fixed schema, PRAGMAs, and sorted insertion order.
9. Runs foreign-key and integrity checks.
10. Builds under a pinned Python and SQLite environment.
11. Emits database size, row counts, source counts, G2P profile counts, and
    SHA-256.

The build does not claim byte determinism across arbitrary SQLite versions. Reproducibility requires the pinned build environment recorded in the manifest.

## GitHub Release publication

A data release such as `lexikos-data-2026.09.1` contains hash-named assets:

```text
observations-<sha256>.jsonl.zst
raw-payloads-<sha256>.tar.zst
curation-<sha256>.sqlite.zst
runtime-<sha256>.sqlite
manifest.json
```

Git contains schema migrations, JSON Schemas, mappings, pack configuration, scraper/importer code, compiler code, and the same manifest.

GitHub permits authorized replacement or deletion of release assets, so immutability is enforced by policy and verification rather than assumed:

- Never overwrite a hash-named asset.
- Retain published data releases.
- Keep ordinary machine or NAS backups of unreleased curation work.
- Exclude or privately publish source payloads whose licenses prohibit public redistribution.
- Split compressed artifacts before GitHub's per-asset size limit when necessary.

## Legacy TSV migration

1. Register every current dictionary as a `source` and one or more `import_run` rows.
2. Record each original TSV file hash, path, parser version `legacy-tsv-v1`, and documented source revision.
3. Import every original physical row and each pronunciation variant as an
   immutable observation with stable row and variant ordinals.
4. Create accepted reviews using the current normalization behavior.
5. Mark migrated file-level metadata with `metadata_origin = "dataset-declaration"`; do not present it as scraped row-level evidence.
6. Leave absent source URL, locality, accent, dialect feature, and license values unknown unless the existing provenance file establishes them.
7. Import each physical Spanish source once. `spa.tsv` can feed `es` and
   `es-es`; `spa-latin.tsv` feeds `es-419`. It must not feed `es-co`, whose
   pack remains empty until an explicitly Colombian source is integrated.
8. Declare one default G2P profile per dictionary-plus-normalizer pack. Use a
   nullable model for Spanish and any other dictionary-only pack.
9. Generate the runtime snapshot and compare supported packs, words, IPA
   values, source multiplicity, G2P profile dictionaries, and documented
   sample lookups against the TSV implementation.
10. Switch both `Lexicon` lookup and `G2p` dictionary/profile lookup to SQLite
    in one cutover.
11. Remove all TSV runtime loaders and packaged TSV copies after parity passes.
    Do not retain a fallback path.

## Failure handling

- Malformed source rows still become observations when their raw fields can be captured; their reviews are rejected with a reason.
- A missing or mismatched raw-payload hash aborts import.
- An unknown or non-redistributable license blocks publication, not local curation.
- Invalid normalized metadata JSON aborts review creation.
- Ambiguous mappings remain explicit and are excluded unless a pack policy deliberately permits them.
- Snapshot generation aborts on foreign-key failures, integrity failures, unknown pack references, active evidence with prohibited redistribution, or nondeterministic duplicate output keys.
- Runtime database format mismatches raise a clear initialization error; there is no silent TSV fallback.
- A dictionary-only G2P miss emits `OOVWarning` and passes the normalized token
  through; it is not a snapshot or model-loading failure.

## Verification strategy

### Curation behavior

- Re-importing the same source row is idempotent.
- Duplicate source occurrences remain distinct observations.
- A correction supersedes rather than mutates an earlier review.
- Rejected and superseded reviews never enter snapshots.
- Identical canonical IPA unions all active accepted evidence.
- Conflicting dialect evidence remains separately attributable.
- Unknown metadata remains unknown.

### Snapshot behavior

- Pack assignment does not relabel source dialects.
- `es-co` has zero pronunciation, evidence, G2P profile, and G2P dictionary rows until an explicitly Colombian source is configured.
- Synthetic evidence follows each pack's explicit policy.
- Fixed inputs in the pinned build environment produce the same manifest counts and SHA-256.
- Foreign-key and integrity checks pass.

### Runtime behavior

- Unfiltered supported-language discovery includes every pack's required
  default profile; filtered discovery includes exact matching model-backed or
  dictionary-only profiles.
- Bare `G2p(lang)` and unfiltered discovery use a non-WikiPron, non-broad
  default profile when configured; explicit selector combinations follow the
  selection table and never silently fall back to the default.
- Known English and Spanish lexicon lookups preserve IPA and provenance
  behavior.
- Both `normalize_phonemes=False` and `True` preserve IPA results; normalized
  collisions union every contributing source.
- Both `include_synthetic=False` and `True` preserve current filtering,
  pronunciation removal, evidence union, and source ordering.
- Every advertised model-backed G2P profile returns the same dictionary-hit
  IPA and unknown-word model output before and after cutover.
- A dictionary-only profile returns dictionary hits normally; an OOV emits
  `OOVWarning` and passes through the normalized token without phoneme
  normalization.
- `G2p(lang)` succeeds for every migrated dictionary-plus-normalizer pack.
- G2P compatibility fixtures cover exact `" ~ "` and comma splitting,
  trimming, `" . "` replacement, repeated-word append order, and last-value
  selection.
- Missing `Lexicon` words raise `KeyError`.
- Mapping methods query the database without loading the complete lexicon.
- Wheel inspection confirms the runtime database and dependency metadata.
- A clean wheel installation successfully performs representative English and Spanish lookups.

## Rollout

1. Add schemas, migrations, pack configuration, importer, and snapshot compiler.
2. Migrate legacy TSVs into a local curation database.
3. Generate and verify the first runtime snapshot.
4. Compare complete pack-level counts and targeted provenance samples.
5. Publish a data release with hashes and licensing boundaries.
6. Replace both the Lexicon and G2P TSV loaders with SQLite-backed lookup.
7. Update package data and public documentation.
8. Remove obsolete TSV runtime assets and code.
9. Build and install the final wheel, then rerun the complete test suite,
   every advertised G2P profile smoke—including dictionary-only OOV
   behavior—and representative English and Spanish lexicon lookups.

The implementation is complete only after the clean cutover; a compiler or database added beside the existing TSV runtime is not considered completion.
