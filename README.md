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

```py
>>> from lexikos import Lexicon
>>> lexicon = Lexicon()
>>> print(lexicon["added"])
{'ˈæ d ə d', 'a d ɪ d', 'æ d ɪ d', 'a d ə d', 'æ ɾ ə d', 'ˈa d ə d', 'æ d ə d', 'ˈæ d ɪ d', 'ˈa d ɪ d', 'ˈæ ɾ ɪ d', 'æ ɾ ɪ d', 'ˈæ ɾ ə d'}
>>> print(lexicon["runner"])
{'ɹ ʌ n ɚ', 'ˈr ʌ n ɝ', 'ɹ ʌ n ə'}
>>> print(lexicon["water"])
{'w ɑ ɾ ɚ', 'ˈw oː t ə', 'w ɑ t ə ɹ', 'w ɔ ɾ ɚ', 'ˈw oː ɾ ə', 'w ɔː t ə', 'ˈw ɔ t ɝ', 'w ɔ t ə ɹ'}
```

## Dictionaries & Models

### English `(en)`

| Language | Dictionary | Phone Set | Corpus                                       | G2P Model |
| -------- | ---------- | --------- | -------------------------------------------- | --------- |
| en       | Wikipron   | IPA       | [Link](./lexikos/dict/wikipron/eng_latn.tsv) |           |

### English `(en-US)`

| Language       | Dictionary   | Phone Set | Corpus                                                                                                                     | G2P Model                                                                                                             |
| -------------- | ------------ | --------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| en-US          | CMU Dict     | ARPA      | [External Link](https://github.com/microsoft/CNTK/blob/master/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train) | [bookbot/byt5-small-cmudict](https://huggingface.co/bookbot/byt5-small-cmudict)                                       |
| en-US          | CMU Dict IPA | IPA       | [External Link](https://github.com/menelik3/cmudict-ipa/blob/master/cmudict-0.7b-ipa.txt)                                  |                                                                                                                       |
| en-US          | CharsiuG2P   | IPA       | [External Link](https://github.com/lingjzhu/CharsiuG2P/blob/main/dicts/eng-us.tsv)                                         | [charsiu/g2p_multilingual_byT5_small_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100)             |
| en-US (Broad)  | Wikipron     | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_us_broad.tsv)                     | [bookbot/byt5-small-wikipron-eng-latn-us-broad](https://huggingface.co/bookbot/byt5-small-wikipron-eng-latn-us-broad) |
| en-US (Narrow) | Wikipron     | IPA       | [External Link](https://github.com/CUNY-CL/wikipron/blob/master/data/scrape/tsv/eng_latn_us_narrow.tsv)                    |

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
| European English       | en-UK, en-HU, en-IE               | United Kingdom, Hungary, Ireland                      |        |           |
| Mexican English        | en-MX                             | Mexico                                                |        |           |
| New Zealand English    | en-NZ                             | New Zealand                                           |        |           |
| North American         | en-CA, en-US                      | Canada, United States                                 |        |           |
| Middle Eastern English | en-EG, en-IL                      | Egypt, Israel                                         |        |           |
| Southeast Asian        | en-TH, en-ID, en-MY, en-PH, en-SG | Thailand, Indonesia, Malaysia, Philippines, Singapore |        |           |
| South Asian English    | en-IN                             | India                                                 |        |           |
  
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