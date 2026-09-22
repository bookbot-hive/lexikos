import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_charsiu_g2p.py"
SPLITS = ("train", "dev", "test")


def _run(source, output, language="es-419"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--language",
            language,
            "--output-dir",
            str(output),
            "--seed",
            "7",
            "--train-ratio",
            "0.6",
            "--dev-ratio",
            "0.2",
        ],
        capture_output=True,
        text=True,
    )


def test_preparation_is_deterministic_and_keeps_words_in_one_split(tmp_path):
    lines = [
        "corazo\u0301n\tkoɾason",
        "corazón\tkoɾaˈson ~ koɾason",
        "niño\tniɲo",
        "hola\tola",
        "ayúdenme\tajudɛn,ajudɛnmɛ",
    ] + ["palabra{}\tsonido{}".format(index, index) for index in range(40)]
    first_source = tmp_path / "first.tsv"
    second_source = tmp_path / "second.tsv"
    first_source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second_source.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    first = _run(first_source, first_output)
    second = _run(second_source, second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    word_splits = {}
    pronunciations_by_word = {}
    pronunciation_count = 0
    for split in SPLITS:
        first_data = (first_output / split / "spa-latin.tsv").read_text(
            encoding="utf-8"
        )
        second_data = (second_output / split / "spa-latin.tsv").read_text(
            encoding="utf-8"
        )
        assert first_data == second_data
        for line in first_data.splitlines():
            word, pronunciation = line.split("\t", 1)
            word_splits.setdefault(word, set()).add(split)
            pronunciations_by_word.setdefault(word, set()).add(pronunciation)
            pronunciation_count += 1

    assert all(len(splits) == 1 for splits in word_splits.values())
    assert "corazón" in word_splits
    assert "corazo\u0301n" not in word_splits
    assert pronunciations_by_word["ayúdenme"] == {"ajudɛn", "ajudɛnmɛ"}
    assert pronunciation_count == 46

    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "charsiu-g2p-tsv-v1"
    assert manifest["language"] == "es-419"
    assert manifest["charsiu_language"] == "spa-latin"
    assert sum(split["words"] for split in manifest["splits"].values()) == 44
    assert sum(split["pronunciations"] for split in manifest["splits"].values()) == 46


def test_preparation_rejects_malformed_dictionary_rows(tmp_path):
    source = tmp_path / "broken.tsv"
    source.write_text("missing-tab\n", encoding="utf-8")

    result = _run(source, tmp_path / "output")

    assert result.returncode != 0
    assert "expected word<TAB>pronunciation" in result.stderr
    assert not (tmp_path / "output" / "manifest.json").exists()


