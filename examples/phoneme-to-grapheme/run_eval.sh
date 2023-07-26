#!/bin/bash


MODEL="bookbot/p2g_charsiu_byt5_tiny_12_layers_100_multi"


python eval.py \
    --model $MODEL \
    --dataset_name bookbot/timit_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/austalk_words_mq_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/l2-arctic_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/libriphone_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/speechocean762_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/sc_cw_children_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common_voice_en_test_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/bookbot_en_v1-v2_v2_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-us_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-gb_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-au_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-nz_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-ca_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

python eval.py \
    --model $MODEL \
    --dataset_name bookbot/common-voice-accent-in_p2g \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --max_length 1024 \
    --batch_size 64 \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” �

