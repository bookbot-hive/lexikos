#!/usr/bin/env python3
"""Build Lexikos curation and runtime SQLite databases from legacy TSV files.

The importer intentionally uses the existing language registry as its only
configuration source.  A dictionary path is imported once into curation even
when several locale packs reference it; runtime compilation then applies each
pack's explicit attribution and synthetic-data policy without changing source
metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lexikos.pronunciations import split_pronunciation_variants
from lexikos.storage import (
    CURATION_SCHEMA_VERSION,
    RUNTIME_FORMAT_VERSION,
    RUNTIME_SCHEMA_VERSION,
    check_curation_integrity,
    check_runtime_integrity,
    open_curation_database,
    open_runtime_database,
)


COMPILER_VERSION = "legacy-tsv-runtime-v2"
PARSER_NAME = "legacy-tsv"
PARSER_VERSION = "legacy-tsv-v1"
MAPPING_VERSION = "legacy-tsv-v1"
EXTRACTION_RULE = "legacy-g2p-v1"
DEFAULT_CONFIG_PATH = "lexikos/languages.py"
DEFAULT_RELEASE_MANIFEST = (
    Path(__file__).resolve().parents[1] / "lexikos" / "data" / "release-manifest.json"
)


@dataclass(frozen=True)
class DictionaryDeclaration:
    path: str
    source_id: str
    source_name: str
    language: str
    pack_id: str
    dialect_json: Optional[str]
    transcription: str
    synthetic: bool
    dictionary: Any
    pack: Any


@dataclass(frozen=True)
class ProfileDeclaration:
    profile: Any
    pack: Any
    dictionary: Any


@dataclass(frozen=True)
class ImportRunInfo:
    id: str
    source_id: str
    source_revision: str
    license_id: Optional[str]
    license_url: Optional[str]
    redistribution_policy: str
    parser_name: str
    parser_version: str
    raw_payload_uri: str
    raw_payload_sha256: str
    started_at: str
    completed_at: str
    path: str


@dataclass(frozen=True)
class BuildResult:
    curation_database: Path
    runtime_database: Path
    curation_sha256: str
    runtime_sha256: str
    import_run_ids: Tuple[str, ...]
    counts: Mapping[str, int]


class BuildError(RuntimeError):
    """Raised when source/configuration data cannot produce a valid snapshot."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _path_key(path: Any) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        value = os.fsdecode(value)
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _dialect_mapping(dialect: Any) -> Optional[Dict[str, Any]]:
    if dialect is None:
        return None
    features: List[Dict[str, str]] = []
    for feature in getattr(dialect, "features", ()) or ():
        features.append(
            {
                "name": str(getattr(feature, "name", "")),
                "value": str(getattr(feature, "value", "")),
            }
        )
    return {
        "territory": getattr(dialect, "territory", None),
        "macroregion": getattr(dialect, "macroregion", None),
        "group": getattr(dialect, "group", None),
        "locality": getattr(dialect, "locality", None),
        "features": features,
    }


def _dialect_json(dialect: Any) -> Optional[str]:
    value = _dialect_mapping(dialect)
    return None if value is None else _canonical_json(value)


def _normalizer_name(normalizer: Any) -> Optional[str]:
    if normalizer is None:
        return None
    return "{}.{}".format(
        getattr(normalizer, "__module__", ""),
        getattr(
            normalizer, "__qualname__", getattr(normalizer, "__name__", "callable")
        ),
    )


def _pack_config(pack: Any) -> Dict[str, Any]:
    dictionaries = []
    for dictionary in getattr(pack, "dictionaries", ()) or ():
        dictionaries.append(
            {
                "path": _path_key(getattr(dictionary, "path", "")),
                "source": getattr(dictionary, "source", ""),
                "dialect": _dialect_mapping(getattr(dictionary, "dialect", None)),
                "transcription": getattr(dictionary, "transcription", "unspecified"),
                "synthetic": bool(getattr(dictionary, "synthetic", False)),
            }
        )
    profiles = []
    for profile in getattr(pack, "g2p_profiles", ()) or ():
        dictionary = getattr(profile, "dictionary", None)
        profiles.append(
            {
                "id": getattr(profile, "id", ""),
                "backend": getattr(profile, "backend", ""),
                "transcription": getattr(profile, "transcription", ""),
                "model": getattr(profile, "model", None),
                "dictionary_import_run_id": getattr(
                    profile, "dictionary_import_run_id", ""
                ),
                "extraction_rule": getattr(profile, "extraction_rule", ""),
                "order": list(getattr(profile, "order", ())),
                "duplicate_word_policy": getattr(profile, "duplicate_word_policy", ""),
                "lookup_selection": getattr(profile, "lookup_selection", ""),
                "dictionary": {
                    "path": _path_key(getattr(dictionary, "path", "")),
                    "source": getattr(dictionary, "source", ""),
                    "dialect": _dialect_mapping(getattr(dictionary, "dialect", None)),
                    "transcription": getattr(
                        dictionary, "transcription", "unspecified"
                    ),
                    "synthetic": bool(getattr(dictionary, "synthetic", False)),
                },
            }
        )
    return {
        "id": getattr(pack, "id", ""),
        "display_name": getattr(pack, "display_name", ""),
        "base_language": getattr(pack, "base_language", ""),
        "territory": getattr(pack, "territory", None),
        "macroregion": getattr(pack, "macroregion", None),
        "dictionaries": dictionaries,
        "g2p_profiles": profiles,
        "default_g2p_profile_id": getattr(pack, "default_g2p_profile_id", ""),
        "text_normalizer": _normalizer_name(getattr(pack, "text_normalizer", None)),
        "phoneme_normalizer": _normalizer_name(
            getattr(pack, "phoneme_normalizer", None)
        ),
    }


def _load_manifest(manifest: Optional[Any]) -> Dict[str, Any]:
    if manifest is None:
        manifest = DEFAULT_RELEASE_MANIFEST
    if isinstance(manifest, Mapping):
        result = dict(manifest)
    else:
        path = Path(manifest)
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BuildError(
                "cannot read release manifest {}: {}".format(path, error)
            ) from error
    if not isinstance(result, dict):
        raise BuildError("release manifest must be a JSON object")
    return result


def _manifest_entry(manifest: Mapping[str, Any], path: str) -> Dict[str, Any]:
    candidates: List[Any] = []
    for key in ("files", "dictionaries", "imports", "import_runs"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            for candidate in (path, path.lstrip("./")):
                if candidate in value and isinstance(value[candidate], Mapping):
                    candidates.append(value[candidate])
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                item_path = item.get(
                    "path", item.get("dictionary_path", item.get("file"))
                )
                if item_path is not None and _path_key(item_path) == path:
                    candidates.append(item)
    # A top-level path map is also convenient for small manifests.
    value = manifest.get(path)
    if isinstance(value, Mapping):
        candidates.append(value)
    result: Dict[str, Any] = {}
    for candidate in candidates:
        result.update(candidate)
    return result


def _resolve_source_file(source_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates = [
        source_root / relative,
        source_root / "lexikos" / "dict" / relative,
        source_root / "dict" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise BuildError(
        "dictionary source {!r} was not found below {}".format(
            relative_path, source_root
        )
    )


def _registry_packs(packs: Optional[Iterable[Any]]) -> Tuple[Any, ...]:
    if packs is not None:
        result = tuple(packs)
    else:
        try:
            from lexikos.languages import supported_lexicon_languages, get_language_pack
        except (
            Exception
        ) as error:  # pragma: no cover - import errors are environment-specific.
            raise BuildError(
                "cannot import the Lexikos language registry: {}".format(error)
            ) from error
        result = tuple(
            get_language_pack(lang) for lang in supported_lexicon_languages()
        )
    if not result:
        raise BuildError("language registry contains no packs")
    ids = [getattr(pack, "id", "") for pack in result]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise BuildError("language pack IDs must be non-empty and unique")
    return tuple(sorted(result, key=lambda pack: getattr(pack, "id", "")))


def _collect_declarations(
    packs: Sequence[Any],
) -> Tuple[Tuple[DictionaryDeclaration, ...], Tuple[ProfileDeclaration, ...]]:
    declarations: Dict[str, DictionaryDeclaration] = {}
    profiles: List[ProfileDeclaration] = []
    profile_ids: Dict[str, str] = {}
    for pack in packs:
        pack_id = str(getattr(pack, "id", ""))
        language = str(getattr(pack, "base_language", ""))
        if not language:
            raise BuildError("pack {!r} has no base language".format(pack_id))
        for dictionary in getattr(pack, "dictionaries", ()) or ():
            path = _path_key(getattr(dictionary, "path", ""))
            source_id = str(getattr(dictionary, "source", ""))
            if not path or not source_id:
                raise BuildError(
                    "pack {!r} contains an incomplete dictionary declaration".format(
                        pack_id
                    )
                )
            declaration = DictionaryDeclaration(
                path=path,
                source_id=source_id,
                source_name=source_id,
                language=language,
                pack_id=pack_id,
                dialect_json=_dialect_json(getattr(dictionary, "dialect", None)),
                transcription=str(getattr(dictionary, "transcription", "unspecified")),
                synthetic=bool(getattr(dictionary, "synthetic", False)),
                dictionary=dictionary,
                pack=pack,
            )
            previous = declarations.get(path)
            if previous is None:
                declarations[path] = declaration
            elif (
                previous.source_id != declaration.source_id
                or previous.language != declaration.language
                or previous.dialect_json != declaration.dialect_json
                or previous.transcription != declaration.transcription
                or previous.synthetic != declaration.synthetic
            ):
                raise BuildError(
                    "dictionary path {!r} has conflicting declarations across packs".format(
                        path
                    )
                )
        for profile in getattr(pack, "g2p_profiles", ()) or ():
            profile_id = str(getattr(profile, "id", ""))
            dictionary = getattr(profile, "dictionary", None)
            path = _path_key(getattr(dictionary, "path", ""))
            if not profile_id or dictionary is None or not path:
                raise BuildError(
                    "pack {!r} contains an incomplete G2P profile".format(pack_id)
                )
            if not getattr(profile, "dictionary_import_run_id", ""):
                raise BuildError(
                    "G2P profile {!r} has no pinned import run".format(profile_id)
                )
            if getattr(profile, "extraction_rule", "") != EXTRACTION_RULE:
                raise BuildError(
                    "G2P profile {!r} has an unsupported extraction rule".format(
                        profile_id
                    )
                )
            if tuple(getattr(profile, "order", ())) != (
                "source_occurrence",
                "variant_occurrence",
            ):
                raise BuildError(
                    "G2P profile {!r} has an unsupported order".format(profile_id)
                )
            if (
                getattr(profile, "duplicate_word_policy", "") != "append"
                or getattr(profile, "lookup_selection", "") != "last"
            ):
                raise BuildError(
                    "G2P profile {!r} has an unsupported duplicate policy".format(
                        profile_id
                    )
                )
            prior_pack = profile_ids.get(profile_id)
            if prior_pack is not None and prior_pack != pack_id:
                raise BuildError(
                    "G2P profile ID {!r} is not globally unique".format(profile_id)
                )
            profile_ids[profile_id] = pack_id
            profiles.append(
                ProfileDeclaration(profile=profile, pack=pack, dictionary=dictionary)
            )
            if path not in declarations:
                source_id = str(getattr(dictionary, "source", ""))
                if not source_id:
                    raise BuildError(
                        "G2P profile {!r} has no dictionary source".format(profile_id)
                    )
                declarations[path] = DictionaryDeclaration(
                    path=path,
                    source_id=source_id,
                    source_name=source_id,
                    language=language,
                    pack_id=pack_id,
                    dialect_json=_dialect_json(getattr(dictionary, "dialect", None)),
                    transcription=str(
                        getattr(dictionary, "transcription", "unspecified")
                    ),
                    synthetic=bool(getattr(dictionary, "synthetic", False)),
                    dictionary=dictionary,
                    pack=pack,
                )
            else:
                declaration = declarations[path]
                if (
                    declaration.source_id != getattr(dictionary, "source", "")
                    or declaration.dialect_json
                    != _dialect_json(getattr(dictionary, "dialect", None))
                    or declaration.transcription
                    != getattr(dictionary, "transcription", "unspecified")
                    or declaration.synthetic
                    != bool(getattr(dictionary, "synthetic", False))
                ):
                    raise BuildError(
                        "G2P profile dictionary {!r} conflicts with its pack declaration".format(
                            path
                        )
                    )
    return tuple(declarations[key] for key in sorted(declarations)), tuple(
        sorted(
            profiles,
            key=lambda item: (
                getattr(item.pack, "id", ""),
                getattr(item.profile, "id", ""),
            ),
        )
    )


def _config_sha256(packs: Sequence[Any]) -> str:
    return _sha256_json(
        [
            _pack_config(pack)
            for pack in sorted(packs, key=lambda p: getattr(p, "id", ""))
        ]
    )


def _manifest_value(
    manifest: Mapping[str, Any], *keys: str, default: Any = None
) -> Any:
    for key in keys:
        if key in manifest and manifest[key] is not None:
            return manifest[key]
    return default


def _required_manifest_text(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise BuildError("release manifest must define non-empty {!r}".format(key))
    return value


def _build_timestamp(manifest: Mapping[str, Any]) -> str:
    return _required_manifest_text(manifest, "build_timestamp")


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise BuildError("cannot read dictionary {}: {}".format(path, error)) from error
    return digest.hexdigest()


def _run_id(source_id: str, path: str, source_revision: str) -> str:
    digest = _sha256_json(
        {
            "kind": "legacy-import",
            "path": path,
            "source_id": source_id,
            "source_revision": source_revision,
        }
    )
    return "legacy-{}-{}".format(source_id, digest[:32])


def _import_run(
    declaration: DictionaryDeclaration,
    source_path: Path,
    manifest: Mapping[str, Any],
    build_timestamp: str,
) -> ImportRunInfo:
    path = declaration.path
    entry = _manifest_entry(manifest, path)
    if not entry:
        raise BuildError("release manifest has no import run for {!r}".format(path))
    file_hash = _source_hash(source_path)
    source_revision = entry.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise BuildError(
            "source revision for {!r} must be a non-empty string".format(path)
        )
    if (
        source_revision.startswith("sha256:")
        and source_revision != "sha256:" + file_hash
    ):
        raise BuildError("source revision hash mismatch for {!r}".format(path))
    raw_hash = entry.get("raw_payload_sha256")
    if not isinstance(raw_hash, str) or not raw_hash:
        raise BuildError(
            "release manifest has no raw payload hash for {!r}".format(path)
        )
    if raw_hash != file_hash:
        raise BuildError("raw payload hash mismatch for {!r}".format(path))
    run_id = entry.get("id", entry.get("import_run_id"))
    if not isinstance(run_id, str) or not run_id:
        raise BuildError("release manifest has no import run ID for {!r}".format(path))
    policy = entry.get(
        "redistribution_policy",
        _manifest_value(manifest, "redistribution_policy", default="unknown"),
    )
    if policy not in {"public", "private", "metadata-only", "unknown"}:
        raise BuildError(
            "invalid redistribution policy {!r} for {!r}".format(policy, path)
        )
    raw_uri = entry.get("raw_payload_uri")
    if not isinstance(raw_uri, str) or not raw_uri:
        raise BuildError(
            "release manifest has no raw payload URI for {!r}".format(path)
        )
    parser_name = entry.get(
        "parser_name", _manifest_value(manifest, "parser_name", default=PARSER_NAME)
    )
    parser_version = entry.get(
        "parser_version",
        _manifest_value(manifest, "parser_version", default=PARSER_VERSION),
    )
    return ImportRunInfo(
        id=run_id,
        source_id=declaration.source_id,
        source_revision=source_revision,
        license_id=entry.get("license_id", _manifest_value(manifest, "license_id")),
        license_url=entry.get("license_url", _manifest_value(manifest, "license_url")),
        redistribution_policy=policy,
        parser_name=str(parser_name),
        parser_version=str(parser_version),
        raw_payload_uri=str(raw_uri),
        raw_payload_sha256=file_hash,
        started_at=str(entry.get("started_at", build_timestamp)),
        completed_at=str(entry.get("completed_at", build_timestamp)),
        path=path,
    )


def _insert_source(
    connection: sqlite3.Connection, declaration: DictionaryDeclaration
) -> None:
    connection.execute(
        "INSERT INTO source(id, name, homepage_url) VALUES (?, ?, ?) ON CONFLICT(id) DO NOTHING",
        (declaration.source_id, declaration.source_name, None),
    )
    row = connection.execute(
        "SELECT name, homepage_url FROM source WHERE id = ?", (declaration.source_id,)
    ).fetchone()
    if (
        row is None
        or row["name"] != declaration.source_name
        or row["homepage_url"] is not None
    ):
        raise BuildError(
            "source {!r} has conflicting metadata".format(declaration.source_id)
        )


def _insert_import_run(connection: sqlite3.Connection, run: ImportRunInfo) -> None:
    values = (
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
    )
    connection.execute(
        """INSERT INTO import_run(
            id, source_id, source_revision, license_id, license_url,
            redistribution_policy, parser_name, parser_version,
            raw_payload_uri, raw_payload_sha256, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING""",
        values,
    )
    row = connection.execute(
        "SELECT * FROM import_run WHERE id = ?", (run.id,)
    ).fetchone()
    if (
        row is None
        or tuple(
            row[key]
            for key in (
                "id",
                "source_id",
                "source_revision",
                "license_id",
                "license_url",
                "redistribution_policy",
                "parser_name",
                "parser_version",
                "raw_payload_uri",
                "raw_payload_sha256",
                "started_at",
                "completed_at",
            )
        )
        != values
    ):
        raise BuildError("import run {!r} has conflicting metadata".format(run.id))


def _observation_id(import_run_id: str, record: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "import_run_id": import_run_id,
            "source_row_reference": record["source_row_reference"],
            "source_occurrence": record["source_occurrence"],
            "variant_occurrence": record["variant_occurrence"],
            "record": record,
        }
    )


def _review_id(observation_id: str, values: Mapping[str, Any]) -> str:
    return _sha256_json({"observation_id": observation_id, "review": values})


def _canonical_legacy_ipa(value: str) -> str:
    import re

    return re.sub(r"\s+", " ", value.replace(".", " ")).strip()


def _legacy_review_metadata() -> str:
    unknown = {
        "raw": None,
        "value": None,
        "rule": None,
        "confidence": None,
        "status": "unknown",
    }
    return _canonical_json(
        {
            "metadata_origin": "dataset-declaration",
            "source_language": dict(unknown),
            "territory": dict(unknown),
            "macroregion": dict(unknown),
            "dialect_group": dict(unknown),
            "locality": dict(unknown),
            "features": [],
        }
    )


def _insert_observation_and_review(
    connection: sqlite3.Connection,
    run: ImportRunInfo,
    declaration: DictionaryDeclaration,
    line_number: int,
    source_occurrence: int,
    word_raw: str,
    pronunciation_raw: str,
    variant_occurrence: int,
    pronunciation_variant_raw: str,
    retrieved_at: str,
    *,
    decision: str = "accepted",
    reason: Optional[str] = None,
) -> Tuple[str, str]:
    if decision not in {"accepted", "rejected"}:
        raise BuildError("unsupported legacy review decision {!r}".format(decision))
    if decision == "rejected" and not reason:
        raise BuildError("rejected legacy review requires a reason")
    metadata_raw_json = _canonical_json({"metadata_origin": "dataset-declaration"})
    source_row_reference = "{}:{}".format(declaration.path, line_number)
    record = {
        "source_url": None,
        "source_row_reference": source_row_reference,
        "source_occurrence": source_occurrence,
        "variant_occurrence": variant_occurrence,
        "word_raw": word_raw,
        "pronunciation_raw": pronunciation_raw,
        "pronunciation_variant_raw": pronunciation_variant_raw,
        "metadata_raw_json": metadata_raw_json,
        "retrieved_at": retrieved_at,
    }
    observation_id = _observation_id(run.id, record)
    observation_values = (
        observation_id,
        run.id,
        None,
        source_row_reference,
        source_occurrence,
        variant_occurrence,
        word_raw,
        pronunciation_raw,
        pronunciation_variant_raw,
        metadata_raw_json,
        retrieved_at,
    )
    connection.execute(
        """INSERT INTO observation(
            id, import_run_id, source_url, source_row_reference,
            source_occurrence, variant_occurrence, word_raw,
            pronunciation_raw, pronunciation_variant_raw,
            metadata_raw_json, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING""",
        observation_values,
    )
    row = connection.execute(
        "SELECT * FROM observation WHERE id = ?", (observation_id,)
    ).fetchone()
    if (
        row is None
        or tuple(
            row[key]
            for key in (
                "id",
                "import_run_id",
                "source_url",
                "source_row_reference",
                "source_occurrence",
                "variant_occurrence",
                "word_raw",
                "pronunciation_raw",
                "pronunciation_variant_raw",
                "metadata_raw_json",
                "retrieved_at",
            )
        )
        != observation_values
    ):
        raise BuildError("observation {!r} has conflicting data".format(observation_id))

    if decision == "accepted":
        language_normalized = declaration.language
        word_normalized = word_raw.lower()
        ipa_normalized = _canonical_legacy_ipa(pronunciation_variant_raw)
    else:
        language_normalized = None
        word_normalized = None
        ipa_normalized = None
    review_values_for_id = {
        "parser_version": run.parser_version,
        "language_normalized": language_normalized,
        "word_normalized": word_normalized,
        "ipa_normalized": ipa_normalized,
        "metadata_normalized_json": _legacy_review_metadata(),
        "mapping_version": MAPPING_VERSION,
        "decision": decision,
        "reason": reason,
        "reviewer": "legacy-migration",
        "created_at": retrieved_at,
        "supersedes_review_id": None,
    }
    review_id = _review_id(observation_id, review_values_for_id)
    review_values = (
        review_id,
        observation_id,
        review_values_for_id["parser_version"],
        language_normalized,
        word_normalized,
        ipa_normalized,
        review_values_for_id["metadata_normalized_json"],
        MAPPING_VERSION,
        decision,
        reason,
        "legacy-migration",
        retrieved_at,
        None,
    )
    connection.execute(
        """INSERT INTO review(
            id, observation_id, parser_version, language_normalized,
            word_normalized, ipa_normalized, metadata_normalized_json,
            mapping_version, decision, reason, reviewer, created_at,
            supersedes_review_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING""",
        review_values,
    )
    row = connection.execute(
        "SELECT * FROM review WHERE id = ?", (review_id,)
    ).fetchone()
    if (
        row is None
        or tuple(
            row[key]
            for key in (
                "id",
                "observation_id",
                "parser_version",
                "language_normalized",
                "word_normalized",
                "ipa_normalized",
                "metadata_normalized_json",
                "mapping_version",
                "decision",
                "reason",
                "reviewer",
                "created_at",
                "supersedes_review_id",
            )
        )
        != review_values
    ):
        raise BuildError("review {!r} has conflicting data".format(review_id))
    if decision == "rejected":
        return observation_id, review_id
    pronunciation_id = _sha256_json(
        {
            "language": language_normalized,
            "normalized_word": word_normalized,
            "normalized_ipa": ipa_normalized,
        }
    )
    connection.execute(
        """INSERT INTO pronunciation(id, language, normalized_word, normalized_ipa)
        VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING""",
        (pronunciation_id, language_normalized, word_normalized, ipa_normalized),
    )
    pronunciation_row = connection.execute(
        "SELECT * FROM pronunciation WHERE id = ?", (pronunciation_id,)
    ).fetchone()
    if pronunciation_row is None or tuple(
        pronunciation_row[key]
        for key in ("id", "language", "normalized_word", "normalized_ipa")
    ) != (pronunciation_id, language_normalized, word_normalized, ipa_normalized):
        raise BuildError(
            "pronunciation {!r} has conflicting data".format(pronunciation_id)
        )
    connection.execute(
        """INSERT INTO pronunciation_evidence(pronunciation_id, accepted_review_id)
        VALUES (?, ?) ON CONFLICT(pronunciation_id, accepted_review_id) DO NOTHING""",
        (pronunciation_id, review_id),
    )
    return observation_id, review_id


def _ingest_dictionary(
    connection: sqlite3.Connection,
    declaration: DictionaryDeclaration,
    source_path: Path,
    run: ImportRunInfo,
    build_timestamp: str,
) -> int:
    count = 0
    try:
        with source_path.open("r", encoding="utf-8", newline="") as file:
            source_occurrence = 0
            for line_number, line in enumerate(file, start=1):
                text = line.rstrip("\r\n")
                if not text.strip():
                    continue
                fields = text.split("\t")
                if len(fields) == 1:
                    word_raw = fields[0]
                    pronunciation_raw = ""
                else:
                    word_raw = fields[0]
                    pronunciation_raw = (
                        fields[1] if len(fields) == 2 else "\t".join(fields[1:])
                    )
                rejection_reason = None
                if len(fields) != 2:
                    rejection_reason = "expected exactly two TSV fields"
                elif not word_raw or not pronunciation_raw:
                    rejection_reason = "expected non-empty word and IPA"
                if rejection_reason is not None:
                    _insert_observation_and_review(
                        connection,
                        run,
                        declaration,
                        line_number,
                        source_occurrence,
                        word_raw,
                        pronunciation_raw,
                        0,
                        pronunciation_raw,
                        build_timestamp,
                        decision="rejected",
                        reason=rejection_reason,
                    )
                    count += 1
                    source_occurrence += 1
                    continue
                try:
                    variants = split_pronunciation_variants(pronunciation_raw)
                except ValueError as error:
                    _insert_observation_and_review(
                        connection,
                        run,
                        declaration,
                        line_number,
                        source_occurrence,
                        word_raw,
                        pronunciation_raw,
                        0,
                        pronunciation_raw,
                        build_timestamp,
                        decision="rejected",
                        reason=str(error),
                    )
                    count += 1
                    source_occurrence += 1
                    continue
                for variant_occurrence, pronunciation_variant_raw in enumerate(
                    variants
                ):
                    _insert_observation_and_review(
                        connection,
                        run,
                        declaration,
                        line_number,
                        source_occurrence,
                        word_raw,
                        pronunciation_raw,
                        variant_occurrence,
                        pronunciation_variant_raw,
                        build_timestamp,
                    )
                    count += 1
                source_occurrence += 1
    except UnicodeDecodeError as error:
        raise BuildError(
            "dictionary {} is not valid UTF-8: {}".format(source_path, error)
        ) from error
    return count


def _ensure_run_source(connection: sqlite3.Connection, run: ImportRunInfo) -> None:
    row = connection.execute(
        "SELECT source_id FROM import_run WHERE id = ?", (run.id,)
    ).fetchone()
    if row is None or row["source_id"] != run.source_id:
        raise BuildError(
            "import run {!r} is not attached to source {!r}".format(
                run.id, run.source_id
            )
        )


def _ingest_curation(
    source_root: Path,
    curation_path: Path,
    declarations: Sequence[DictionaryDeclaration],
    manifest: Mapping[str, Any],
    build_timestamp: str,
) -> Tuple[Tuple[ImportRunInfo, ...], Dict[str, int]]:
    connection = open_curation_database(curation_path)
    runs: List[ImportRunInfo] = []
    counts: Dict[str, int] = {"observations": 0, "reviews": 0, "pronunciations": 0}
    try:
        with connection:
            for declaration in declarations:
                source_path = _resolve_source_file(source_root, declaration.path)
                run = _import_run(declaration, source_path, manifest, build_timestamp)
                _insert_source(connection, declaration)
                existing = connection.execute(
                    "SELECT 1 FROM import_run WHERE id = ?", (run.id,)
                ).fetchone()
                _insert_import_run(connection, run)
                _ensure_run_source(connection, run)
                if existing is None:
                    before_observations = connection.execute(
                        "SELECT COUNT(*) FROM observation"
                    ).fetchone()[0]
                    _ingest_dictionary(
                        connection, declaration, source_path, run, build_timestamp
                    )
                    after_observations = connection.execute(
                        "SELECT COUNT(*) FROM observation"
                    ).fetchone()[0]
                    counts["observations"] += after_observations - before_observations
                runs.append(run)
            counts["reviews"] = connection.execute(
                "SELECT COUNT(*) FROM review"
            ).fetchone()[0]
            counts["pronunciations"] = connection.execute(
                "SELECT COUNT(*) FROM pronunciation"
            ).fetchone()[0]
        check_curation_integrity(connection)
    finally:
        connection.close()
    return tuple(runs), counts


def _terminal_accepted_reviews(connection: sqlite3.Connection) -> List[sqlite3.Row]:
    rows = connection.execute(
        """SELECT
            r.id AS review_id,
            r.observation_id,
            r.language_normalized,
            r.word_normalized,
            r.ipa_normalized,
            r.metadata_normalized_json,
            r.decision,
            o.import_run_id,
            o.source_url,
            o.source_row_reference,
            o.source_occurrence,
            o.variant_occurrence,
            o.word_raw,
            o.pronunciation_raw,
            o.pronunciation_variant_raw,
            i.source_id,
            i.source_revision,
            i.license_id,
            i.license_url,
            i.redistribution_policy,
            s.name AS source_name
        FROM review AS r
        JOIN observation AS o ON o.id = r.observation_id
        JOIN import_run AS i ON i.id = o.import_run_id
        JOIN source AS s ON s.id = i.source_id
        WHERE r.decision = 'accepted'
          AND NOT EXISTS (
              SELECT 1 FROM review AS newer
              WHERE newer.supersedes_review_id = r.id
          )
        ORDER BY o.import_run_id, o.source_occurrence, o.variant_occurrence, r.id"""
    ).fetchall()
    return list(rows)


def _source_language_values(row: sqlite3.Row) -> Tuple[Optional[str], Optional[str]]:
    try:
        metadata = json.loads(row["metadata_normalized_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise BuildError(
            "review {!r} has invalid normalized metadata".format(row["review_id"])
        ) from error
    if not isinstance(metadata, dict):
        raise BuildError(
            "review {!r} normalized metadata is not an object".format(row["review_id"])
        )
    field = metadata.get("source_language")
    if isinstance(field, dict):
        return field.get("value"), field.get("raw")
    if isinstance(field, str):
        return field, field
    return None, None


def _runtime_attribution(
    declaration: DictionaryDeclaration,
    pack: Any,
    row: sqlite3.Row,
) -> Dict[str, Any]:
    source_language, source_language_raw = _source_language_values(row)
    return {
        "source_id": row["source_id"],
        "source_name": row["source_name"],
        "language": getattr(pack, "id", ""),
        "source_language": source_language,
        "source_language_raw": source_language_raw,
        "source_url": row["source_url"],
        "source_revision": row["source_revision"],
        "license_id": row["license_id"],
        "license_url": row["license_url"],
        "evidence_status": "accepted",
        "dialect_json": declaration.dialect_json,
        "transcription": declaration.transcription,
        "synthetic": int(declaration.synthetic),
    }


def _runtime_profile_policy(
    profile: Any, dictionary: Any, import_run_id: str
) -> Dict[str, Any]:
    declared_run_id = str(getattr(profile, "dictionary_import_run_id", ""))
    if declared_run_id != import_run_id:
        raise BuildError(
            "G2P profile {!r} pins import run {!r}, not {!r}".format(
                getattr(profile, "id", ""), declared_run_id, import_run_id
            )
        )
    extraction_rule = getattr(profile, "extraction_rule", "")
    order = tuple(getattr(profile, "order", ()))
    duplicate_word_policy = getattr(profile, "duplicate_word_policy", "")
    lookup_selection = getattr(profile, "lookup_selection", "")
    if extraction_rule != EXTRACTION_RULE:
        raise BuildError("unsupported G2P extraction rule {!r}".format(extraction_rule))
    if order != ("source_occurrence", "variant_occurrence"):
        raise BuildError("unsupported G2P dictionary order {!r}".format(order))
    if duplicate_word_policy != "append" or lookup_selection != "last":
        raise BuildError("unsupported G2P duplicate/lookup policy")
    return {
        "review_filter": "terminal-accepted",
        "include_synthetic": bool(getattr(dictionary, "synthetic", False)),
        "extraction_rule": extraction_rule,
        "order": list(order),
        "duplicate_word_policy": duplicate_word_policy,
        "lookup_selection": lookup_selection,
        "source_id": getattr(dictionary, "source", ""),
        "import_run_id": declared_run_id,
        "transcription": getattr(profile, "transcription", ""),
    }


def _runtime_pronunciation_ipa(pack: Any, ipa: str) -> Optional[str]:
    normalizer = getattr(pack, "phoneme_normalizer", None)
    if normalizer is None:
        return None
    value = normalizer(ipa)
    if not isinstance(value, str):
        raise BuildError(
            "phoneme normalizer for pack {!r} did not return text".format(
                getattr(pack, "id", "")
            )
        )
    return value


def _insert_runtime_metadata(
    connection: sqlite3.Connection, metadata: Mapping[str, Any]
) -> None:
    for key in sorted(metadata):
        value = metadata[key]
        if not isinstance(value, str):
            value = _canonical_json(value)
        connection.execute(
            "INSERT INTO snapshot_metadata(key, value) VALUES (?, ?)", (key, value)
        )


def _runtime_database(
    curation_path: Path,
    runtime_path: Path,
    packs: Sequence[Any],
    declarations: Sequence[DictionaryDeclaration],
    profiles: Sequence[ProfileDeclaration],
    runs: Sequence[ImportRunInfo],
    manifest: Mapping[str, Any],
    config_sha256: str,
) -> Dict[str, int]:
    run_by_path = {run.path: run for run in runs}
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    curation = open_curation_database(curation_path, readonly=True, create=False)
    try:
        check_curation_integrity(curation)
        accepted = _terminal_accepted_reviews(curation)
        for row in accepted:
            if (
                row["language_normalized"] is None
                or row["word_normalized"] is None
                or row["ipa_normalized"] is None
            ):
                raise BuildError(
                    "accepted review {!r} is not normalized".format(row["review_id"])
                )
            link = curation.execute(
                """SELECT 1
                FROM pronunciation_evidence AS pe
                JOIN pronunciation AS p ON p.id = pe.pronunciation_id
                WHERE pe.accepted_review_id = ?
                  AND p.language = ?
                  AND p.normalized_word = ?
                  AND p.normalized_ipa = ?""",
                (
                    row["review_id"],
                    row["language_normalized"],
                    row["word_normalized"],
                    row["ipa_normalized"],
                ),
            ).fetchone()
            if link is None:
                raise BuildError(
                    "accepted review {!r} has no canonical evidence link".format(
                        row["review_id"]
                    )
                )
    finally:
        curation.close()
    current_run_ids = {run.id for run in runs}
    accepted_by_run: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in accepted:
        if row["import_run_id"] not in current_run_ids:
            continue
        if row["redistribution_policy"] != "public":
            raise BuildError(
                "active evidence from import run {!r} has non-public redistribution policy {!r}".format(
                    row["import_run_id"], row["redistribution_policy"]
                )
            )
        accepted_by_run[row["import_run_id"]].append(row)
    runtime = open_runtime_database(runtime_path, readonly=False, create=True)
    try:
        with runtime:
            pronunciation_groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for pack in packs:
                pack_id = str(getattr(pack, "id", ""))
                language = str(getattr(pack, "base_language", ""))
                for dictionary in getattr(pack, "dictionaries", ()) or ():
                    path = _path_key(getattr(dictionary, "path", ""))
                    declaration = declaration_by_path.get(path)
                    run = run_by_path.get(path)
                    if declaration is None or run is None:
                        raise BuildError(
                            "pack {!r} references an unimported dictionary {!r}".format(
                                pack_id, path
                            )
                        )
                    if run.source_id != getattr(dictionary, "source", ""):
                        raise BuildError(
                            "dictionary source mismatch for {!r}".format(path)
                        )
                    for row in accepted_by_run.get(run.id, ()):
                        if row["source_id"] != run.source_id:
                            raise BuildError(
                                "source mismatch in import run {!r}".format(run.id)
                            )
                        if row["language_normalized"] != language:
                            raise BuildError(
                                "accepted review {!r} language does not match pack {!r}".format(
                                    row["review_id"], pack_id
                                )
                            )
                        word = row["word_normalized"]
                        ipa = row["ipa_normalized"]
                        if word is None or ipa is None:
                            raise BuildError(
                                "accepted review {!r} is not normalized".format(
                                    row["review_id"]
                                )
                            )
                        group_key = (pack_id, word, ipa)
                        group = pronunciation_groups.setdefault(
                            group_key,
                            {
                                "pack_id": pack_id,
                                "language": language,
                                "normalized_word": word,
                                "ipa": ipa,
                                "phoneme_normalized_ipa": _runtime_pronunciation_ipa(
                                    pack, ipa
                                ),
                                "evidence": {},
                            },
                        )
                        attribution = _runtime_attribution(declaration, pack, row)
                        attribution_key = _canonical_json(attribution)
                        evidence_group = group["evidence"].setdefault(
                            attribution_key,
                            {"attribution": attribution, "observation_ids": set()},
                        )
                        evidence_group["observation_ids"].add(row["observation_id"])
            # Every configured dictionary is represented, even if all its rows
            # were rejected; packs with no active pronunciation rows simply have
            # no runtime pronunciation record.
            pronunciation_keys = sorted(pronunciation_groups)
            pronunciation_ids: Dict[Tuple[str, str, str], int] = {}
            for identifier, key in enumerate(pronunciation_keys, start=1):
                pronunciation_ids[key] = identifier
                group = pronunciation_groups[key]
                runtime.execute(
                    """INSERT INTO pronunciation(
                        id, pack_id, language, normalized_word, ipa,
                        phoneme_normalized_ipa
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        identifier,
                        group["pack_id"],
                        group["language"],
                        group["normalized_word"],
                        group["ipa"],
                        group["phoneme_normalized_ipa"],
                    ),
                )
            evidence_count = 0
            for key in pronunciation_keys:
                group = pronunciation_groups[key]
                pronunciation_id = pronunciation_ids[key]
                for attribution_key in sorted(group["evidence"]):
                    evidence_group = group["evidence"][attribution_key]
                    observation_ids = tuple(sorted(evidence_group["observation_ids"]))
                    identity = {
                        "pronunciation": {
                            "pack_id": group["pack_id"],
                            "language": group["language"],
                            "normalized_word": group["normalized_word"],
                            "ipa": group["ipa"],
                        },
                        "attribution": evidence_group["attribution"],
                        "observation_ids": observation_ids,
                    }
                    evidence_id = _sha256_json(identity)
                    attribution = evidence_group["attribution"]
                    runtime.execute(
                        """INSERT INTO evidence(
                            id, pronunciation_id, source_id, source_name,
                            language, source_language, source_language_raw,
                            observation_ids_json, source_url, source_revision,
                            license_id, license_url, evidence_status,
                            dialect_json, transcription, synthetic
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evidence_id,
                            pronunciation_id,
                            attribution["source_id"],
                            attribution["source_name"],
                            attribution["language"],
                            attribution["source_language"],
                            attribution["source_language_raw"],
                            _canonical_json(list(observation_ids)),
                            attribution["source_url"],
                            attribution["source_revision"],
                            attribution["license_id"],
                            attribution["license_url"],
                            attribution["evidence_status"],
                            attribution["dialect_json"],
                            attribution["transcription"],
                            attribution["synthetic"],
                        ),
                    )
                    evidence_count += 1

            profile_count = 0
            dictionary_count = 0
            default_by_pack: Dict[str, int] = defaultdict(int)
            for profile_declaration in profiles:
                profile = profile_declaration.profile
                pack = profile_declaration.pack
                dictionary = profile_declaration.dictionary
                pack_id = str(getattr(pack, "id", ""))
                profile_id = str(getattr(profile, "id", ""))
                path = _path_key(getattr(dictionary, "path", ""))
                run = run_by_path.get(path)
                if run is None:
                    raise BuildError(
                        "G2P profile {!r} references an unimported dictionary".format(
                            profile_id
                        )
                    )
                if run.source_id != getattr(dictionary, "source", ""):
                    raise BuildError(
                        "G2P profile {!r} source mismatch".format(profile_id)
                    )
                default_id = getattr(pack, "default_g2p_profile_id", None)
                is_default = int(default_id == profile_id)
                default_by_pack[pack_id] += is_default
                policy = _runtime_profile_policy(profile, dictionary, run.id)
                runtime.execute(
                    """INSERT INTO g2p_profile(
                        id, pack_id, backend, transcription, model, is_default,
                        dictionary_source_id, dictionary_import_run_id,
                        extraction_rule, config_path, config_sha256,
                        dictionary_policy_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        profile_id,
                        pack_id,
                        str(getattr(profile, "backend", "")),
                        str(getattr(profile, "transcription", "")),
                        getattr(profile, "model", None),
                        is_default,
                        str(getattr(dictionary, "source", "")),
                        run.id,
                        str(getattr(profile, "extraction_rule", "")),
                        str(
                            getattr(
                                pack,
                                "config_path",
                                _manifest_value(
                                    manifest, "config_path", default=DEFAULT_CONFIG_PATH
                                ),
                            )
                        ),
                        config_sha256,
                        _canonical_json(policy),
                    ),
                )
                profile_count += 1
                profile_rows = list(accepted_by_run.get(run.id, ()))
                profile_rows.sort(
                    key=lambda row: (
                        row["source_occurrence"],
                        row["variant_occurrence"],
                        row["observation_id"],
                    )
                )
                seen_positions: Dict[Tuple[int, int], str] = {}
                ordinals_by_word: Dict[str, int] = defaultdict(int)
                for row in profile_rows:
                    position = (row["source_occurrence"], row["variant_occurrence"])
                    if position in seen_positions:
                        raise BuildError(
                            "nondeterministic duplicate G2P dictionary key for {!r} at {}".format(
                                profile_id, position
                            )
                        )
                    seen_positions[position] = row["review_id"]
                    try:
                        variants = split_pronunciation_variants(
                            row["pronunciation_raw"]
                        )
                    except ValueError as error:
                        raise BuildError(
                            "invalid G2P pronunciation for {!r}".format(
                                row["review_id"]
                            )
                        ) from error
                    variant_index = row["variant_occurrence"]
                    if (
                        variant_index >= len(variants)
                        or variants[variant_index] != row["pronunciation_variant_raw"]
                    ):
                        raise BuildError(
                            "G2P variant occurrence mismatch for {!r}".format(
                                row["review_id"]
                            )
                        )
                    lookup_word = row["word_raw"].lower()
                    output_ipa = row["pronunciation_variant_raw"].replace(" . ", " ")
                    ordinal = ordinals_by_word[lookup_word]
                    ordinals_by_word[lookup_word] = ordinal + 1
                    runtime.execute(
                        """INSERT INTO g2p_dictionary(
                            profile_id, lookup_word, ordinal, output_ipa,
                            observation_id, accepted_review_id
                        ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            profile_id,
                            lookup_word,
                            ordinal,
                            output_ipa,
                            row["observation_id"],
                            row["review_id"],
                        ),
                    )
                    dictionary_count += 1
            for pack in packs:
                pack_id = str(getattr(pack, "id", ""))
                profile_list = [
                    item for item in profiles if getattr(item.pack, "id", "") == pack_id
                ]
                requires_default = bool(getattr(pack, "dictionaries", ())) or bool(
                    profile_list
                )
                requires_default = (
                    requires_default
                    and getattr(pack, "text_normalizer", None) is not None
                )
                if requires_default and (
                    not profile_list or default_by_pack[pack_id] != 1
                ):
                    raise BuildError(
                        "pack {!r} must declare exactly one default G2P profile".format(
                            pack_id
                        )
                    )

            input_hashes = {
                run.path: run.raw_payload_sha256
                for run in sorted(runs, key=lambda r: r.path)
            }
            source_counts = defaultdict(int)
            for declaration in declarations:
                source_counts[declaration.source_id] += 1
            metadata = {
                "format_version": RUNTIME_FORMAT_VERSION,
                "runtime_schema_version": str(RUNTIME_SCHEMA_VERSION),
                "curation_schema_version": str(CURATION_SCHEMA_VERSION),
                "compiler_version": str(
                    _manifest_value(
                        manifest, "compiler_version", default=COMPILER_VERSION
                    )
                ),
                "parser_versions": _canonical_json(
                    sorted({run.parser_version for run in runs})
                ),
                "mapping_version": MAPPING_VERSION,
                "config_commit": _required_manifest_text(manifest, "config_commit"),
                "config_sha256": config_sha256,
                "build_timestamp": _build_timestamp(manifest),
                "python_version": platform_python_version(),
                "sqlite_version": sqlite3.sqlite_version,
                "included_import_run_ids": _canonical_json(
                    [run.id for run in sorted(runs, key=lambda r: r.id)]
                ),
                "input_hashes": _canonical_json(input_hashes),
                "pack_ids": _canonical_json(
                    sorted(getattr(pack, "id", "") for pack in packs)
                ),
                "counts": _canonical_json(
                    {
                        "pronunciations": len(pronunciation_keys),
                        "evidence": evidence_count,
                        "g2p_profiles": profile_count,
                        "g2p_dictionary": dictionary_count,
                    }
                ),
                "source_counts": _canonical_json(dict(sorted(source_counts.items()))),
            }
            _insert_runtime_metadata(runtime, metadata)
        check_runtime_integrity(runtime)
    finally:
        runtime.close()
    return {
        "pronunciations": len(pronunciation_groups),
        "evidence": evidence_count,
        "g2p_profiles": profile_count,
        "g2p_dictionary": dictionary_count,
    }


def platform_python_version() -> str:
    # A short, stable value is enough for snapshot provenance and does not
    # encode a process path or local build flags.
    return "{}.{}.{}".format(
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro
    )


def _sha256_file(path: Path) -> str:
    return _source_hash(path)


def build_databases(
    source_root: Any,
    curation_database: Any,
    runtime_database: Any,
    *,
    release_manifest: Optional[Any] = None,
    packs: Optional[Iterable[Any]] = None,
) -> BuildResult:
    """Import configured TSVs and atomically publish a runtime snapshot."""
    source_root_path = Path(source_root).expanduser().resolve()
    curation_path = Path(curation_database).expanduser().resolve()
    runtime_path = Path(runtime_database).expanduser().resolve()
    if curation_path == runtime_path:
        raise BuildError("curation and runtime database paths must differ")
    manifest = _load_manifest(release_manifest)
    build_timestamp = _build_timestamp(manifest)
    environment = manifest.get("build_environment")
    if not isinstance(environment, Mapping):
        raise BuildError("release manifest must define build_environment")
    if environment.get("python_version") != platform_python_version():
        raise BuildError(
            "release manifest Python version does not match this interpreter"
        )
    if environment.get("sqlite_version") != sqlite3.sqlite_version:
        raise BuildError(
            "release manifest SQLite version does not match this interpreter"
        )
    if manifest.get("compiler_version") != COMPILER_VERSION:
        raise BuildError(
            "release manifest compiler version does not match this compiler"
        )
    registry = _registry_packs(packs)
    declarations, profile_declarations = _collect_declarations(registry)
    config_sha256 = _config_sha256(registry)

    curation_is_existing = curation_path.exists()
    curation_temporary = curation_path.with_name(".{}.tmp".format(curation_path.name))
    if not curation_is_existing and curation_temporary.exists():
        curation_temporary.unlink()
    curation_target = curation_path if curation_is_existing else curation_temporary
    runtime_temporary = runtime_path.with_name(".{}.tmp".format(runtime_path.name))
    if runtime_temporary.exists():
        runtime_temporary.unlink()
    try:
        runs, curation_counts = _ingest_curation(
            source_root_path,
            curation_target,
            declarations,
            manifest,
            build_timestamp,
        )
        runtime_counts = _runtime_database(
            curation_target,
            runtime_temporary,
            registry,
            declarations,
            profile_declarations,
            runs,
            manifest,
            config_sha256,
        )
        expected_runtime_hash = manifest.get("runtime_sha256")
        if expected_runtime_hash is not None:
            actual_runtime_hash = _sha256_file(runtime_temporary)
            if expected_runtime_hash != actual_runtime_hash:
                raise BuildError(
                    "runtime snapshot hash mismatch: expected {}, got {}".format(
                        expected_runtime_hash, actual_runtime_hash
                    )
                )
        if not curation_is_existing:
            os.replace(str(curation_temporary), str(curation_path))
        os.replace(str(runtime_temporary), str(runtime_path))
    except Exception:
        if curation_temporary.exists() and not curation_is_existing:
            curation_temporary.unlink()
        if runtime_temporary.exists():
            runtime_temporary.unlink()
        raise
    return BuildResult(
        curation_database=curation_path,
        runtime_database=runtime_path,
        curation_sha256=_sha256_file(curation_path),
        runtime_sha256=_sha256_file(runtime_path),
        import_run_ids=tuple(run.id for run in sorted(runs, key=lambda run: run.id)),
        counts={**curation_counts, **runtime_counts},
    )


def _default_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "lexikos" / "dict"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_default_source_root(),
        help="root containing the configured DictionarySource paths",
    )
    parser.add_argument(
        "--curation-db",
        "--curation-output",
        dest="curation_database",
        type=Path,
        default=Path("curation.sqlite3"),
        help="curation SQLite output (existing files are imported idempotently)",
    )
    parser.add_argument(
        "--runtime-db",
        "--runtime-output",
        dest="runtime_database",
        type=Path,
        default=Path("runtime.sqlite3"),
        help="runtime SQLite output (rebuilt from scratch)",
    )
    parser.add_argument(
        "--release-manifest",
        "--manifest",
        dest="release_manifest",
        type=Path,
        default=DEFAULT_RELEASE_MANIFEST,
        help="pinned JSON release manifest (default: lexikos/data/release-manifest.json)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_databases(
            arguments.source_root,
            arguments.curation_database,
            arguments.runtime_database,
            release_manifest=arguments.release_manifest,
        )
    except (BuildError, OSError, sqlite3.Error) as error:
        raise SystemExit("build failed: {}".format(error)) from error
    print(
        _canonical_json(
            {
                "curation_database": str(result.curation_database),
                "runtime_database": str(result.runtime_database),
                "curation_sha256": result.curation_sha256,
                "runtime_sha256": result.runtime_sha256,
                "import_run_ids": result.import_run_ids,
                "counts": dict(result.counts),
            }
        )
    )
    return 0


if __name__ == "__main__":
    main()
