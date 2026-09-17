# Lexikos - λεξικός /lek.si.kós/

A collection of pronunciation dictionaries and neural grapheme-to-phoneme models.

<p align="center">
    <img src="https://github.com/bookbot-hive/lexikos/raw/main/assets/lexikos.png" alt="logo" width="300"/>
</p>

## Install Lexikos

Install from PyPI

```sh
pip install lexikos
```

Editable install from Source

```sh
git clone https://github.com/bookbot-hive/lexikos.git
pip install -e lexikos
```

## Usage

Language selection is explicit. Language IDs are exact, lowercase identifiers;
for example, use `en-us`, not `en_US` or `EN-US`.

```py
>>> from lexikos import G2p, Lexicon
>>> Lexicon.supported_languages()
('en', 'en-au', 'en-ca', 'en-in', 'en-nz', 'en-uk', 'en-us', 'es', 'es-419', 'es-co', 'es-es', 'es-mx')
>>> G2p.supported_languages()
('en', 'en-au', 'en-ca', 'en-in', 'en-nz', 'en-uk', 'en-us')
```

### Lexicon

`Lexicon[word]` returns one immutable `Pronunciation` per unique IPA value.
Each pronunciation retains every dictionary, language, dialect, transcription
classification, and synthetic-data source that supplies that IPA.

```py
>>> from lexikos import Lexicon
>>> lexicon = Lexicon("en-us")
>>> pronunciation = next(p for p in lexicon["a"] if p.ipa == "ə")
>>> pronunciation.ipa
'ə'
>>> [(source.source, source.language, source.dialect.group) for source in pronunciation.sources]
[('cmudict', 'en-us', 'american'), ('librispeech', 'en-us', 'american'), ('wikipron', 'en-us', 'american')]
```

Spanish lexical packs are available for generic Spanish, Spain, Latin America,
Mexico, and Colombia. Generic `es` currently uses the same explicitly
peninsular source as `es-es`, so it is not dialect-neutral. The Colombia pack
uses the explicitly Latin-American CharsiuG2P source rather than relabeling it
as country-specific evidence:

```py
>>> lexicon = Lexicon("es-co")
>>> [(p.ipa, p.sources[0].dialect.macroregion) for p in lexicon["niño"]]
[('niɲo', 'latin-america')]
```

When normalization collapses multiple IPA strings, their source records are
unioned rather than discarded:

```py
>>> lexicon = Lexicon("en-us", normalize_phonemes=True)
```

Synthetic pronunciations remain opt-in and are limited to datasets explicitly
assigned to the selected language pack:

```py
>>> lexicon = Lexicon("en-us", include_synthetic=True)
```

### Phonemization

`G2p` also requires an explicit language. Only complete language packs with a
dictionary, broad G2P model, and text normalizer are selectable.

```py
>>> from lexikos import G2p
>>> g2p = G2p("en-us")
>>> g2p("Hello there! $100 is not a lot of money in 2023.")
['h ɛ l o ʊ', 'ð ɛ ə ɹ', 'w ʌ n', 'h ʌ n d ɹ ɪ d', 'd ɑ l ɚ z', 'ɪ z', 'n ɒ t', 'ə', 'l ɑ t', 'ʌ v', 'm ʌ n i', 'ɪ n', 't w ɛ n t i', 't w ɛ n t i', 'θ ɹ iː']
>>> g2p = G2p("en-au")
>>> g2p("Hi there mate! Have a g'day!")
['h a ɪ', 'θ ɛ ə ɹ', 'm e ɪ t', 'h e ɪ v', 'ə', 'ɡ ə ˈd æ ɪ']
```

Spanish neural G2P is not advertised by `G2p.supported_languages()` until a
compatible model artifact exists. Use `charsiu_prompt` to construct the exact
multilingual CharsiuG2P input when fine-tuning or serving such a model:

```py
>>> from lexikos import charsiu_prompt
>>> charsiu_prompt("es-co", "NIÑO")
'<spa-co>: niño'
```

## Dictionaries & Models

### English `(en)`

| Language | Dictionary | Phone Set | Corpus                                       | G2P Model                                                                                           |
| -------- | ---------- | --------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| en       | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn.tsv) | [bookbot/byt5-small-wikipron-eng-latn](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn) |

### English `(en-US)`

| Language       | Dictionary   | Phone Set | Corpus                                                                                                                     | G2P Model                                                                                                             |
| -------------- | ------------ | --------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| en-US          | CMU Dict     | ARPA      | [External Link](https://github.com/microsoft/CNTK/blob/master/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train) | [bookbot/byt5-small-cmudict](https://huggingface.co/bookbot/byt5-small-cmudict)                                       |
| en-US          | CMU Dict IPA | IPA       | [External Link](https://github.com/menelik3/cmudict-ipa/blob/master/cmudict-0.7b-ipa.txt)                                  |                                                                                                                       |
| en-US          | CharsiuG2P   | IPA       | [External Link](https://github.com/lingjzhu/CharsiuG2P/blob/main/dicts/eng-us.tsv)                                         | [charsiu/g2p_multilingual_byT5_small_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100)             |
| en-US (Broad)  | Wikipron     | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_us_broad.tsv)                     | [bookbot/byt5-small-wikipron-eng-latn-us-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-us-broad) |
| en-US (Narrow) | Wikipron     | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_us_narrow.tsv)                    |
| en-US          | LibriSpeech  | IPA       | [Link](./lexikos/dict/cmudict-ipa/librispeech-lexicon-200k-allothers-g2p-ipa.tsv)                                          |                                                                                                                       |

### English `(en-UK)`

| Language       | Dictionary | Phone Set | Corpus                                                                                                  | G2P Model                                                                                                             |
| -------------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| en-UK          | CharsiuG2P | IPA       | [External Link](https://github.com/lingjzhu/CharsiuG2P/blob/main/dicts/eng-uk.tsv)                      | [charsiu/g2p_multilingual_byT5_small_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100)             |
| en-UK (Broad)  | Wikipron   | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_uk_broad.tsv)  | [bookbot/byt5-small-wikipron-eng-latn-uk-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-uk-broad) |
| en-UK (Narrow) | Wikipron   | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_uk_narrow.tsv) |                                                                                                                       |

### English `(en-AU)`

| Language       | Dictionary | Phone Set | Corpus                                                 | G2P Model                                                                                                             |
| -------------- | ---------- | --------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| en-AU (Broad)  | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_au_broad.tsv)  | [bookbot/byt5-small-wikipron-eng-latn-au-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-au-broad) |
| en-AU (Narrow) | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_au_narrow.tsv) |                                                                                                                       |
| en-AU          | AusTalk    | IPA       | [Link](./lexikos/dict/asr-data/austalk_en_au.tsv)      |                                                                                                                       |
| en-AU          | SC-CW      | IPA       | [Link](./lexikos/dict/asr-data/sc_cw_en_au.tsv)        |                                                                                                                       |

### English `(en-CA)`

| Language       | Dictionary | Phone Set | Corpus                                                 | G2P Model                                                                                                             |
| -------------- | ---------- | --------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| en-CA (Broad)  | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_ca_broad.tsv)  | [bookbot/byt5-small-wikipron-eng-latn-ca-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-ca-broad) |
| en-CA (Narrow) | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_ca_narrow.tsv) |                                                                                                                       |

### English `(en-NZ)`

| Language       | Dictionary | Phone Set | Corpus                                                 | G2P Model                                                                                                             |
| -------------- | ---------- | --------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| en-NZ (Broad)  | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_nz_broad.tsv)  | [bookbot/byt5-small-wikipron-eng-latn-nz-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-nz-broad) |
| en-NZ (Narrow) | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_nz_narrow.tsv) |                                                                                                                       |

### English `(en-IN)`

| Language       | Dictionary | Phone Set | Corpus                                                 | G2P Model                                                                                                             |
| -------------- | ---------- | --------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| en-IN (Broad)  | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_in_broad.tsv)  | [bookbot/byt5-small-wikipron-eng-latn-in-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-in-broad) |
| en-IN (Narrow) | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn_in_narrow.tsv) |                                                                                                                       |


### Spanish

| Lexikos language | CharsiuG2P tag | Dictionary |
| ---------------- | -------------- | ---------- |
| `es`             | `spa`          | `spa.tsv` |
| `es-es`          | `spa`          | `spa.tsv` |
| `es-419`         | `spa-latin`    | `spa-latin.tsv` |
| `es-mx`          | `spa-me`       | `spa-me.tsv` |
| `es-co`          | `spa-co`       | `spa-latin.tsv` (Latin-American source) |

The dictionaries are pinned to CharsiuG2P revision
`0c929390759fb94f8ecdfc05cc0bc5f2ff2dc0f4`; provenance and licensing are in
[`lexikos/dict/charsiu`](./lexikos/dict/charsiu/).

## Preparing Spanish data for CharsiuG2P

The preparation CLI accepts Lexikos `word<TAB>IPA` files, expands both ` ~ `
and comma-separated pronunciation variants, normalizes Unicode to NFC,
deduplicates exact pairs, and assigns every pronunciation of a word to one
deterministic split:

```sh
python scripts/prepare_charsiu_g2p.py \
    lexikos/dict/charsiu/spa-latin.tsv \
    --language es-419 \
    --output-dir prepared/es-419
```

This writes headerless CharsiuG2P inputs at
`prepared/es-419/{train,dev,test}/spa-latin.tsv` plus a manifest containing
source and output SHA-256 hashes. Fine-tune with the upstream trainer:

The pinned upstream trainer omits the required space after its language prefix.
Before training, update both prefix expressions in its `src/data_utils.py` from
`'<'+language+'>:' + word` to `'<'+language+'>: ' + word`. Without this fix,
the checkpoint is trained on a different input format from `charsiu_prompt`.

```sh
python /path/to/CharsiuG2P/src/train.py \
    --train \
    --language spa-latin \
    --train_data prepared/es-419/train/spa-latin.tsv \
    --dev_data prepared/es-419/dev/spa-latin.tsv \
    --model byt5 \
    --model_name charsiu/g2p_multilingual_byT5_small_100 \
    --pretrained_model True \
    --output_dir models/spanish-latin
```

## Training G2P Model

We modified the sequence-to-sequence training script of [🤗 HuggingFace](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py) for the purpose of training G2P models. Refer to their [installation requirements](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) for more details.

Training a new G2P model generally follow this recipe:

```diff
python run_translation.py \
+   --model_name_or_path $PRETRAINED_MODEL \
+   --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_strategy epoch \
    --max_source_length 64 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --pad_to_max_length True \
    --overwrite_output_dir \
    --do_train --do_eval \
    --bf16 \
    --predict_with_generate \
    --report_to tensorboard \
    --push_to_hub \
+   --hub_model_id $HUB_MODEL_ID \
    --use_auth_token
```

### Example: Fine-tune ByT5 on CMU Dict

```sh
python run_translation.py \
    --model_name_or_path google/byt5-small \
    --dataset_name bookbot/cmudict-0.7b \
    --output_dir ./byt5-small-cmudict \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_strategy epoch \
    --max_source_length 64 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --pad_to_max_length True \
    --overwrite_output_dir \
    --do_train --do_eval \
    --bf16 \
    --predict_with_generate \
    --report_to tensorboard \
    --push_to_hub \
    --hub_model_id bookbot/byt5-small-cmudict \
    --use_auth_token
```

## Evaluating G2P Model

Then to evaluate:

```diff
python eval.py \
+   --model $PRETRAINED_MODEL \
+   --dataset_name $DATASET_NAME \
    --source_text_column_name source \
    --target_text_column_name target \
    --max_length 64 \
    --batch_size 64
```

### Example: Evaluate ByT5 on CMU Dict

```sh
python eval.py \
    --model bookbot/byt5-small-cmudict \
    --dataset_name bookbot/cmudict-0.7b \
    --source_text_column_name source \
    --target_text_column_name target \
    --max_length 64 \
    --batch_size 64
```

## Corpus Roadmap

### Wikipron

| Language Family        | Code                              | Region                                                | Corpus | G2P Model |
| ---------------------- | --------------------------------- | ----------------------------------------------------- | :----: | :-------: |
| African English        | en-ZA                             | South Africa                                          |        |           |
| Australian English     | en-AU                             | Australia                                             |   ✅    |     ✅     |
| East Asian English     | en-CN, en-HK, en-JP, en-KR, en-TW | China, Hong Kong, Japan, South Korea, Taiwan          |        |           |
| European English       | en-UK, en-HU, en-IE               | United Kingdom, Hungary, Ireland                      |   🚧    |     🚧     |
| Mexican English        | en-MX                             | Mexico                                                |        |           |
| New Zealand English    | en-NZ                             | New Zealand                                           |   ✅    |     ✅     |
| North American         | en-CA, en-US                      | Canada, United States                                 |   ✅    |     ✅     |
| Middle Eastern English | en-EG, en-IL                      | Egypt, Israel                                         |        |           |
| Southeast Asian        | en-TH, en-ID, en-MY, en-PH, en-SG | Thailand, Indonesia, Malaysia, Philippines, Singapore |        |           |
| South Asian English    | en-IN                             | India                                                 |   ✅    |     ✅     |
  
## Resources

- [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- [Microsoft CNTK](https://github.com/microsoft/CNTK/tree/master)
- [CMU Pronouncing Dictionary - IPA](https://github.com/menelik3/cmudict-ipa)
- [Wikipron](https://github.com/CUNY-CL/wikipron/tree/master)

## References

```bibtex
@inproceedings{lee-etal-2020-massively,
    title = "Massively Multilingual Pronunciation Modeling with {W}iki{P}ron",
    author = "Lee, Jackson L.  and
      Ashby, Lucas F.E.  and
      Garza, M. Elizabeth  and
      Lee-Sikka, Yeonju  and
      Miller, Sean  and
      Wong, Alan  and
      McCarthy, Arya D.  and
      Gorman, Kyle",
    booktitle = "Proceedings of LREC",
    year = "2020",
    publisher = "European Language Resources Association",
    pages = "4223--4228",
}
```

```bibtex
@misc{zhu2022byt5,
    title={ByT5 model for massively multilingual grapheme-to-phoneme conversion}, 
    author={Jian Zhu and Cong Zhang and David Jurgens},
    year={2022},
    eprint={2204.03067},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```