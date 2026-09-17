"""SQLite schemas and connection helpers for Lexikos data artifacts.

The curation database is a local, append-friendly record of legacy imports and
review history.  The runtime database is a compact, read-only snapshot consumed
by the public APIs.  This module deliberately contains no project-specific
normalization or registry logic; the snapshot compiler owns that policy.
"""

from __future__ import annotations

from pathlib import Path
import os
import sqlite3
from typing import Optional, Union
from urllib.parse import quote


CURATION_SCHEMA_VERSION = 1
RUNTIME_SCHEMA_VERSION = 1
CURATION_FORMAT_VERSION = "curation-v1"
RUNTIME_FORMAT_VERSION = "runtime-v1"

# The package artifact imported by Lexicon and G2p.  Build tooling writes this
# file (or an explicitly supplied destination) and the runtime opens it read
# only.  Keeping this path here avoids each consumer inventing a package-data
# location.
RUNTIME_DATABASE = Path(__file__).resolve().parent / "data" / "runtime.sqlite3"


CURATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    homepage_url TEXT
);

CREATE TABLE IF NOT EXISTS import_run (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES source(id),
    source_revision TEXT NOT NULL,
    license_id TEXT,
    license_url TEXT,
    redistribution_policy TEXT NOT NULL
        CHECK (redistribution_policy IN ('public', 'private', 'metadata-only', 'unknown')),
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    raw_payload_uri TEXT NOT NULL,
    raw_payload_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation (
    id TEXT PRIMARY KEY,
    import_run_id TEXT NOT NULL REFERENCES import_run(id),
    source_url TEXT,
    source_row_reference TEXT NOT NULL,
    source_occurrence INTEGER NOT NULL CHECK (source_occurrence >= 0),
    variant_occurrence INTEGER NOT NULL CHECK (variant_occurrence >= 0),
    word_raw TEXT NOT NULL,
    pronunciation_raw TEXT NOT NULL,
    pronunciation_variant_raw TEXT NOT NULL,
    metadata_raw_json TEXT NOT NULL
        CHECK (json_valid(metadata_raw_json) = 1
               AND json_type(metadata_raw_json) = 'object'),
    retrieved_at TEXT NOT NULL,
    UNIQUE (import_run_id, source_occurrence, variant_occurrence)
);

CREATE TABLE IF NOT EXISTS review (
    id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES observation(id),
    parser_version TEXT NOT NULL,
    language_normalized TEXT,
    word_normalized TEXT,
    ipa_normalized TEXT,
    metadata_normalized_json TEXT NOT NULL
        CHECK (json_valid(metadata_normalized_json) = 1
               AND json_type(metadata_normalized_json) = 'object'),
    mapping_version TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('accepted', 'rejected', 'superseded')),
    reason TEXT,
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    supersedes_review_id TEXT UNIQUE REFERENCES review(id)
);

CREATE TABLE IF NOT EXISTS pronunciation (
    id TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    normalized_ipa TEXT NOT NULL,
    UNIQUE (language, normalized_word, normalized_ipa)
);

CREATE TABLE IF NOT EXISTS pronunciation_evidence (
    pronunciation_id TEXT NOT NULL REFERENCES pronunciation(id),
    accepted_review_id TEXT NOT NULL REFERENCES review(id),
    PRIMARY KEY (pronunciation_id, accepted_review_id)
);

CREATE INDEX IF NOT EXISTS idx_import_run_source ON import_run(source_id);
CREATE INDEX IF NOT EXISTS idx_observation_import_order
    ON observation(import_run_id, source_occurrence, variant_occurrence);
CREATE INDEX IF NOT EXISTS idx_review_observation ON review(observation_id);
CREATE INDEX IF NOT EXISTS idx_review_terminal_scan ON review(supersedes_review_id, decision);
CREATE INDEX IF NOT EXISTS idx_pronunciation_lookup
    ON pronunciation(language, normalized_word);
CREATE INDEX IF NOT EXISTS idx_evidence_review ON pronunciation_evidence(accepted_review_id);

-- Observations and reviews are immutable historical records.  INSERT OR IGNORE
-- remains available for idempotent migration, while accidental edits cannot
-- silently rewrite provenance.
CREATE TRIGGER IF NOT EXISTS observation_append_only_update
BEFORE UPDATE ON observation
BEGIN
    SELECT RAISE(ABORT, 'observation rows are append-only');
END;

CREATE TRIGGER IF NOT EXISTS observation_append_only_delete
BEFORE DELETE ON observation
BEGIN
    SELECT RAISE(ABORT, 'observation rows are append-only');
END;

CREATE TRIGGER IF NOT EXISTS review_append_only_update
BEFORE UPDATE ON review
BEGIN
    SELECT RAISE(ABORT, 'review rows are append-only');
END;

CREATE TRIGGER IF NOT EXISTS review_append_only_delete
BEFORE DELETE ON review
BEGIN
    SELECT RAISE(ABORT, 'review rows are append-only');
END;
"""


RUNTIME_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pronunciation (
    id INTEGER PRIMARY KEY,
    pack_id TEXT NOT NULL,
    language TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    ipa TEXT NOT NULL,
    phoneme_normalized_ipa TEXT,
    UNIQUE (pack_id, normalized_word, ipa)
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    pronunciation_id INTEGER NOT NULL REFERENCES pronunciation(id),
    source_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    language TEXT NOT NULL,
    source_language TEXT,
    source_language_raw TEXT,
    observation_ids_json TEXT NOT NULL
        CHECK (json_valid(observation_ids_json) = 1
               AND json_type(observation_ids_json) = 'array'
               AND json_array_length(observation_ids_json) > 0),
    source_url TEXT,
    source_revision TEXT NOT NULL,
    license_id TEXT,
    license_url TEXT,
    evidence_status TEXT NOT NULL,
    dialect_json TEXT
        CHECK (dialect_json IS NULL
               OR (json_valid(dialect_json) = 1 AND json_type(dialect_json) = 'object')),
    transcription TEXT NOT NULL,
    synthetic INTEGER NOT NULL CHECK (synthetic IN (0, 1))
);

CREATE TABLE IF NOT EXISTS g2p_profile (
    id TEXT PRIMARY KEY,
    pack_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    transcription TEXT NOT NULL,
    model TEXT,
    is_default INTEGER NOT NULL CHECK (is_default IN (0, 1)),
    dictionary_source_id TEXT NOT NULL,
    dictionary_import_run_id TEXT NOT NULL,
    extraction_rule TEXT NOT NULL,
    config_path TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    dictionary_policy_json TEXT NOT NULL
        CHECK (json_valid(dictionary_policy_json) = 1
               AND json_type(dictionary_policy_json) = 'object')
);

CREATE TABLE IF NOT EXISTS g2p_dictionary (
    profile_id TEXT NOT NULL REFERENCES g2p_profile(id),
    lookup_word TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    output_ipa TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    accepted_review_id TEXT NOT NULL,
    PRIMARY KEY (profile_id, lookup_word, ordinal)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_g2p_profile_default
    ON g2p_profile(pack_id) WHERE is_default = 1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_g2p_profile_backend_transcription
    ON g2p_profile(pack_id, backend, transcription);
CREATE INDEX IF NOT EXISTS idx_runtime_evidence_pronunciation
    ON evidence(pronunciation_id);
"""

# Names kept as aliases for callers that prefer an explicit SQL suffix.
CURATION_SCHEMA_SQL = CURATION_SCHEMA
RUNTIME_SCHEMA_SQL = RUNTIME_SCHEMA

PathLike = Union[str, os.PathLike, Path]


def _connect(
    path: PathLike, *, readonly: bool = False, create: bool = True
) -> sqlite3.Connection:
    """Open a SQLite path and enable the invariants required by both stores."""
    database_path = Path(path).expanduser()
    if readonly:
        if not database_path.exists():
            raise FileNotFoundError(str(database_path))
        # quote() leaves slash safe only when safe is supplied explicitly; URI
        # parsing still receives an absolute path and therefore handles spaces.
        uri_path = quote(str(database_path.resolve()), safe="/")
        connection = sqlite3.connect("file:{}?mode=ro".format(uri_path), uri=True)
    else:
        if not create and not database_path.exists():
            raise FileNotFoundError(str(database_path))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if readonly:
        connection.execute("PRAGMA query_only = ON")
    return connection


def _schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _set_schema_version(connection: sqlite3.Connection, version: int) -> None:
    connection.execute("PRAGMA user_version = {}".format(int(version)))


def _ensure_tables(connection: sqlite3.Connection, schema: str, version: int) -> None:
    current = _schema_version(connection)
    if current == version:
        return
    if current not in (0, version):
        raise sqlite3.DatabaseError(
            "unsupported SQLite schema version {}; expected {}".format(current, version)
        )
    connection.executescript(schema)
    _set_schema_version(connection, version)
    connection.commit()


def create_curation_schema(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Create or verify the curation schema on an existing connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    _ensure_tables(connection, CURATION_SCHEMA, CURATION_SCHEMA_VERSION)
    return connection


def create_runtime_schema(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Create or verify the runtime schema on an existing connection."""
    connection.execute("PRAGMA foreign_keys = ON")
    _ensure_tables(connection, RUNTIME_SCHEMA, RUNTIME_SCHEMA_VERSION)
    return connection


def open_curation_database(
    path: PathLike, *, readonly: bool = False, create: bool = True
) -> sqlite3.Connection:
    """Open a curation database, optionally creating its schema."""
    connection = _connect(path, readonly=readonly, create=create)
    try:
        if create and not readonly:
            create_curation_schema(connection)
        else:
            check_schema_version(connection, CURATION_SCHEMA_VERSION)
    except Exception:
        connection.close()
        raise
    return connection


def open_runtime_database(
    path: Optional[PathLike] = None, *, readonly: bool = True, create: bool = False
) -> sqlite3.Connection:
    """Open the packaged runtime snapshot.

    With no path, ``lexikos/data/runtime.sqlite3`` is opened read-only.  The
    optional arguments make the same helper useful to build tooling and to
    consumers that keep a snapshot outside the wheel.
    """
    database_path = RUNTIME_DATABASE if path is None else path
    connection = _connect(database_path, readonly=readonly, create=create)
    try:
        if create and not readonly:
            create_runtime_schema(connection)
        else:
            check_schema_version(connection, RUNTIME_SCHEMA_VERSION)
    except Exception:
        connection.close()
        raise
    return connection


def check_schema_version(connection: sqlite3.Connection, expected: int) -> None:
    """Raise ``DatabaseError`` when a database is not the expected format."""
    actual = _schema_version(connection)
    if actual != expected:
        raise sqlite3.DatabaseError(
            "unsupported SQLite schema version {}; expected {}".format(actual, expected)
        )


def check_integrity(connection: sqlite3.Connection) -> None:
    """Run SQLite's foreign-key and b-tree integrity checks."""
    foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise sqlite3.DatabaseError(
            "foreign key check failed: {}".format(
                "; ".join(str(tuple(row)) for row in foreign_key_errors)
            )
        )
    result = connection.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        raise sqlite3.DatabaseError(
            "integrity check failed: {}".format(result[0] if result else "no result")
        )


def check_curation_integrity(connection: sqlite3.Connection) -> None:
    check_schema_version(connection, CURATION_SCHEMA_VERSION)
    check_integrity(connection)


def check_runtime_integrity(connection: sqlite3.Connection) -> None:
    check_schema_version(connection, RUNTIME_SCHEMA_VERSION)
    check_integrity(connection)


def close_database(connection: Optional[sqlite3.Connection]) -> None:
    """Close a connection when present; convenient for error paths."""
    if connection is not None:
        connection.close()


__all__ = [
    "CURATION_FORMAT_VERSION",
    "CURATION_SCHEMA",
    "CURATION_SCHEMA_SQL",
    "CURATION_SCHEMA_VERSION",
    "RUNTIME_DATABASE",
    "RUNTIME_FORMAT_VERSION",
    "RUNTIME_SCHEMA",
    "RUNTIME_SCHEMA_SQL",
    "RUNTIME_SCHEMA_VERSION",
    "check_curation_integrity",
    "check_integrity",
    "check_runtime_integrity",
    "check_schema_version",
    "close_database",
    "create_curation_schema",
    "create_runtime_schema",
    "open_curation_database",
    "open_runtime_database",
]
