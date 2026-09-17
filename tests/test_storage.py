import importlib.util
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from lexikos.storage import open_curation_database, open_runtime_database


_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "lexikos_build_databases",
    Path(__file__).resolve().parents[1] / "scripts" / "build_lexicon_databases.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
sys.modules[_BUILDER_SPEC.name] = _BUILDER
_BUILDER_SPEC.loader.exec_module(_BUILDER)
DictionaryDeclaration = _BUILDER.DictionaryDeclaration
ImportRunInfo = _BUILDER.ImportRunInfo
_ingest_dictionary = _BUILDER._ingest_dictionary


@pytest.mark.parametrize("opener", [open_curation_database, open_runtime_database])
def test_create_false_does_not_create_missing_database(tmp_path, opener):
    database = tmp_path / "missing.sqlite3"

    with pytest.raises(FileNotFoundError):
        opener(database, readonly=False, create=False)

    assert not database.exists()


def test_reopening_existing_schema_does_not_mutate_database(tmp_path):
    database = tmp_path / "curation.sqlite3"
    connection = open_curation_database(database)
    connection.close()
    original = database.read_bytes()

    reopened = open_curation_database(database)
    reopened.close()

    assert database.read_bytes() == original


def test_current_version_schema_recreates_missing_tables(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    repaired = open_runtime_database(database, readonly=False, create=True)
    try:
        assert (
            repaired.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'g2p_profile'"
            ).fetchone()
            is not None
        )
    finally:
        repaired.close()


def test_readonly_database_enforces_query_only(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    writable = open_runtime_database(database, readonly=False, create=True)
    writable.close()

    readonly = open_runtime_database(database, readonly=True, create=False)
    try:
        assert readonly.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            readonly.execute(
                "INSERT INTO snapshot_metadata(key, value) VALUES ('key', 'value')"
            )
    finally:
        readonly.close()


def test_malformed_rows_are_preserved_as_rejected_observations(tmp_path):
    source_root = tmp_path / "source"
    source_file = source_root / "fixture.tsv"
    source_root.mkdir()
    source_file.write_text(
        "good\tg ʊ d\nmissing\t\n\tp\nextra\ta\tb\nbroken\ta,\nno-tab\n",
        encoding="utf-8",
    )
    database = tmp_path / "curation.sqlite3"
    connection = open_curation_database(database)
    run = ImportRunInfo(
        id="fixture-run",
        source_id="fixture-source",
        source_revision="fixture-revision",
        license_id="MIT",
        license_url="https://example.invalid/license",
        redistribution_policy="public",
        parser_name="legacy-tsv",
        parser_version="legacy-tsv-v1",
        raw_payload_uri="fixture://fixture.tsv",
        raw_payload_sha256="0" * 64,
        started_at="2026-09-17T00:00:00Z",
        completed_at="2026-09-17T00:00:00Z",
        path="fixture.tsv",
    )
    declaration = DictionaryDeclaration(
        path="fixture.tsv",
        source_id=run.source_id,
        source_name="Fixture",
        language="en",
        pack_id="en",
        dialect_json=None,
        transcription="broad",
        synthetic=False,
        dictionary=SimpleNamespace(),
        pack=SimpleNamespace(),
    )
    try:
        with connection:
            connection.execute(
                "INSERT INTO source(id, name, homepage_url) VALUES (?, ?, ?)",
                (run.source_id, run.source_id, None),
            )
            connection.execute(
                """INSERT INTO import_run(
                    id, source_id, source_revision, license_id, license_url,
                    redistribution_policy, parser_name, parser_version,
                    raw_payload_uri, raw_payload_sha256, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    run.source_id,
                    run.source_revision,
                    run.license_id,
                    run.license_url,
                    run.redistribution_policy,
                    run.parser_name,
                    run.parser_version,
                    run.raw_payload_uri,
                    run.raw_payload_sha256,
                    run.started_at,
                    run.completed_at,
                ),
            )
            assert (
                _ingest_dictionary(
                    connection,
                    declaration,
                    source_file,
                    run,
                    run.completed_at,
                )
                == 6
            )

        decisions = connection.execute(
            "SELECT decision, reason FROM review ORDER BY decision, reason"
        ).fetchall()
        assert [row["decision"] for row in decisions].count("accepted") == 1
        assert [row["decision"] for row in decisions].count("rejected") == 5
        assert all(row["reason"] for row in decisions if row["decision"] == "rejected")
        observations = connection.execute(
            """SELECT source_occurrence, word_raw, pronunciation_raw,
                pronunciation_variant_raw
            FROM observation ORDER BY source_occurrence"""
        ).fetchall()
        assert [tuple(row) for row in observations] == [
            (0, "good", "g ʊ d", "g ʊ d"),
            (1, "missing", "", ""),
            (2, "", "p", "p"),
            (3, "extra", "a\tb", "a\tb"),
            (4, "broken", "a,", "a,"),
            (5, "no-tab", "", ""),
        ]
        assert (
            connection.execute("SELECT count(*) FROM pronunciation").fetchone()[0] == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM pronunciation_evidence"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()
