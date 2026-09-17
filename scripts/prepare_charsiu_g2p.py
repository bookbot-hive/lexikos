#!/usr/bin/env python3
"""Prepare Lexikos dictionaries for CharsiuG2P fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Dict, Iterable, Optional, Sequence, Set

from lexikos.charsiu import CHARSIU_LANGUAGE_TAGS
from lexikos.pronunciations import split_pronunciation_variants


SPLITS = ("train", "dev", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_entries(paths: Iterable[Path]) -> Dict[str, Set[str]]:
    entries: Dict[str, Set[str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                if line.count("\t") != 1:
                    raise ValueError(
                        "{}:{}: expected word<TAB>pronunciation".format(
                            path, line_number
                        )
                    )
                raw_word, raw_pronunciations = line.split("\t")
                word = unicodedata.normalize("NFC", raw_word.strip()).casefold()
                pronunciations = {
                    unicodedata.normalize("NFC", pronunciation)
                    for pronunciation in split_pronunciation_variants(
                        raw_pronunciations
                    )
                }
                if not word or not pronunciations or "" in pronunciations:
                    raise ValueError(
                        "{}:{}: word and pronunciation must be non-empty".format(
                            path, line_number
                        )
                    )
                entries.setdefault(word, set()).update(pronunciations)
    if not entries:
        raise ValueError("no pronunciation entries found")
    return entries


def _split_for_word(word: str, seed: int, train_ratio: float, dev_ratio: float) -> str:
    key = "{}\0{}".format(seed, word).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + dev_ratio:
        return "dev"
    return "test"


def prepare(
    inputs: Sequence[Path],
    language: str,
    output_dir: Path,
    seed: int = 13,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> Path:
    if language not in CHARSIU_LANGUAGE_TAGS:
        raise ValueError("unsupported language {!r}".format(language))
    if not 0 < train_ratio < 1:
        raise ValueError("train ratio must be between 0 and 1")
    if dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("train and dev ratios must leave a non-empty test ratio")

    entries = _read_entries(inputs)
    tag = CHARSIU_LANGUAGE_TAGS[language]
    split_entries = {split: [] for split in SPLITS}
    split_words = {split: set() for split in SPLITS}

    for word in sorted(entries):
        split = _split_for_word(word, seed, train_ratio, dev_ratio)
        split_words[split].add(word)
        split_entries[split].extend(
            (word, pronunciation) for pronunciation in sorted(entries[word])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / "{}.tsv".format(tag)
        output_path.write_text(
            "".join(
                "{}\t{}\n".format(word, pronunciation)
                for word, pronunciation in split_entries[split]
            ),
            encoding="utf-8",
        )
        output_paths[split] = output_path

    manifest = {
        "format": "charsiu-g2p-tsv-v1",
        "language": language,
        "charsiu_language": tag,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "test": round(1 - train_ratio - dev_ratio, 12),
        },
        "sources": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
        "splits": {
            split: {
                "path": str(output_paths[split].relative_to(output_dir)),
                "words": len(split_words[split]),
                "pronunciations": len(split_entries[split]),
                "sha256": _sha256(output_paths[split]),
            }
            for split in SPLITS
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Lexikos word<TAB>IPA dictionaries into deterministic, "
            "word-disjoint CharsiuG2P train/dev/test files."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--language", required=True, choices=tuple(CHARSIU_LANGUAGE_TAGS)
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest_path = prepare(
            inputs=args.inputs,
            language=args.language,
            output_dir=args.output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
