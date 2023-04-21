python eval.py \
    --model bookbot/byt5-small-cmudict \
    --dataset_name bookbot/cmudict-0.7b \
    --source_text_column_name source \
    --target_text_column_name target \
    --max_length 64 \
    --batch_size 64