# Structured Lexicon Storage Design

**Date:** 2026-09-17  
**Status:** Proposed; approved in chat for written review  
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

A scraper or importer emits one JSONL row per source occurrence before database import. The transport schema includes:

```json
{
  "source_id": "wikipron",
  "source_revision": "upstream-revision-or-snapshot-id",
  "source_url": "https://example.test/entry",
  "source_row_reference": "page-or-row-identifier",
  "source_occurrence": 42,
  "word_raw": "niño",
  "pronunciation_raw": "/ˈniɲo/",
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
| `source_occurrence` | INTEGER | NOT NULL, unique within one import run and row reference |
| `word_raw` | TEXT | NOT NULL |
| `pronunciation_raw` | TEXT | NOT NULL |
| `metadata_raw_json` | TEXT | NOT NULL, valid JSON object |
| `retrieved_at` | TEXT | NOT NULL ISO-8601 UTC |

The observation ID is SHA-256 over the import-run ID, source row reference,
source occurrence, and canonical bytes of the raw record. Importers assign a
stable ordinal or byte offset to `source_occurrence`. This preserves duplicate
identical rows even when the upstream source repeats the same row reference.

Unique constraint: `(import_run_id, source_row_reference, source_occurrence)`.

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
| `pronunciation_id` | TEXT | REFERENCES `pronunciation(id)` |
| `accepted_review_id` | TEXT | REFERENCES `review(id)` |

Primary key: `(pronunciation_id, accepted_review_id)`.

Historical links remain in curation. The snapshot compiler includes only evidence whose review is currently terminal and accepted. If a correction changes the normalized word or IPA, the new review links to the new canonical pronunciation; the old pronunciation disappears from runtime when it has no active evidence.

## Curation invariants and merge semantics

1. Raw payloads, JSONL rows, observations, and reviews are append-only.
2. Normalization never changes `word_raw`, `pronunciation_raw`, or `metadata_raw_json`.
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
- Optional G2P backend, transcription, model, and dictionary ordering.

Pack configuration selects accepted evidence; it never changes its dialect metadata. `es-co` may select pooled Latin-American evidence only when the pack explicitly declares that fallback. The resulting `PronunciationSource.dialect` remains pooled Latin-American, not Colombian.

## Runtime SQLite schema

The package ships one database containing all locale packs to avoid duplicating shared evidence.

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
| `normalized_ipa` | TEXT | NOT NULL |

Unique constraint: `(pack_id, normalized_word, normalized_ipa)`. Lookup index: `(pack_id, normalized_word)`.

### `evidence`

One row per unique accepted attribution record for a runtime pronunciation:

| Column | Type |
| --- | --- |
| `pronunciation_id` | INTEGER REFERENCES `pronunciation(id)` |
| `source_id` | TEXT |
| `source_name` | TEXT |
| `language` | TEXT |
| `source_language` | TEXT |
| `source_language_raw` | TEXT |
| `observation_ids_json` | TEXT |
| `source_url` | TEXT |
| `source_revision` | TEXT |
| `license_id` | TEXT |
| `license_url` | TEXT |
| `evidence_status` | TEXT |
| `dialect_json` | TEXT |
| `transcription` | TEXT |
| `synthetic` | INTEGER |

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
| `model` | TEXT | NOT NULL |

Unique constraint: `(pack_id, backend, transcription)`.

### `g2p_dictionary`

The exact dictionary view used before neural fallback:

| Column | Type | Constraint |
| --- | --- | --- |
| `profile_id` | TEXT | REFERENCES `g2p_profile(id)` |
| `normalized_word` | TEXT | NOT NULL |
| `ordinal` | INTEGER | NOT NULL |
| `normalized_ipa` | TEXT | NOT NULL |

Primary key: `(profile_id, normalized_word, ordinal)`. A lookup index covers
`(profile_id, normalized_word)`. `ordinal` preserves the configured source
ordering needed for behavioral parity with the current dictionary loader; the
highest matching ordinal is the selected dictionary pronunciation.

## Runtime API

The public lookup shape remains:

```python
lexicon = Lexicon("es-mx")
pronunciations = lexicon["niño"]
```

`Pronunciation` remains an immutable IPA plus sources. `PronunciationSource` gains:

```text
source_id
observation_ids
source_url
license_id
license_url
source_revision
evidence_status
source_language
source_language_raw
```

Existing fields remain:

```text
source
language
dialect
transcription
synthetic
```

`Lexicon` becomes a read-only mapping backed by indexed SQLite queries rather than loading every row into a `UserDict`. `__getitem__`, `__contains__`, `__iter__`, and `__len__` preserve mapping behavior. Missing words continue to raise `KeyError`.

The runtime opens the packaged database read-only. It exposes accepted provenance only. Curation history requires the separate curation artifact and tooling.

`G2p` reads its profile, dictionary hits, and model identifier from
`g2p_profile` and `g2p_dictionary` in the same runtime snapshot. A known-word
hit returns the configured final pronunciation by `ordinal`; an unknown word
uses the profile's model. No G2P code reads packaged TSV files after cutover.
Locale packs without a configured model, including all Spanish packs in the
initial cutover, have no `g2p_profile` row and are not advertised as G2P
profiles.

## Deterministic snapshot generation

The compiler:

1. Reads a Git-tracked release manifest naming exact import-run IDs.
2. Verifies all raw-payload hashes and database integrity.
3. Resolves terminal accepted reviews.
4. Applies the locale-pack configuration at the recorded Git commit.
5. Groups canonical pronunciations and attribution-equivalent evidence.
6. Compiles reviewed G2P profile metadata and ordered dictionary views.
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
- Reject any asset whose bytes differ from the Git-tracked hash.
- Keep ordinary machine or NAS backups of unreleased curation work.
- Exclude or privately publish source payloads whose licenses prohibit public redistribution.
- Split compressed artifacts before GitHub's per-asset size limit when necessary.

## Legacy TSV migration

1. Register every current dictionary as a `source` and one or more `import_run` rows.
2. Record each original TSV file hash, path, parser version `legacy-tsv-v1`, and documented source revision.
3. Import every original row and pronunciation variant as an immutable observation.
4. Create accepted reviews using the current normalization behavior.
5. Mark migrated file-level metadata with `metadata_origin = "dataset-declaration"`; do not present it as scraped row-level evidence.
6. Leave absent source URL, locality, accent, dialect feature, and license values unknown unless the existing provenance file establishes them.
7. Import each physical Spanish source once. `spa.tsv` can feed `es` and `es-es`; `spa-latin.tsv` can feed `es-419` and explicitly configured `es-co`; pack assignment must not duplicate or relabel evidence.
8. Generate the runtime snapshot and compare supported packs, words, IPA
   values, source multiplicity, G2P profile dictionaries, and documented
   sample lookups against the TSV implementation.
9. Switch both `Lexicon` lookup and `G2p` dictionary/profile lookup to SQLite in
   one cutover.
10. Remove all TSV runtime loaders and packaged TSV copies after parity passes.
    Do not retain a fallback path.

## Failure handling

- Malformed source rows still become observations when their raw fields can be captured; their reviews are rejected with a reason.
- A missing or mismatched raw-payload hash aborts import.
- An unknown or non-redistributable license blocks publication, not local curation.
- Invalid normalized metadata JSON aborts review creation.
- Ambiguous mappings remain explicit and are excluded unless a pack policy deliberately permits them.
- Snapshot generation aborts on foreign-key failures, integrity failures, unknown pack references, active evidence with prohibited redistribution, or nondeterministic duplicate output keys.
- Runtime database format mismatches raise a clear initialization error; there is no silent TSV fallback.

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
- `es-co` pooled evidence remains Latin-American in returned metadata.
- Synthetic evidence follows each pack's explicit policy.
- Fixed inputs in the pinned build environment produce the same manifest counts and SHA-256.
- Foreign-key and integrity checks pass.

### Runtime behavior

- Existing supported-language discovery remains stable unless deliberately changed by pack configuration.
- Known English and Spanish lookups preserve IPA and provenance behavior.
- Every advertised English G2P profile returns the same dictionary-hit IPA
  before and after cutover.
- English G2P unknown-word fallback loads the same model profile and preserves
  observable output for a fixed smoke input.
- Missing words raise `KeyError`.
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
   every advertised English G2P profile smoke, and representative English and
   Spanish lexicon lookups.

The implementation is complete only after the clean cutover; a compiler or database added beside the existing TSV runtime is not considered completion.
