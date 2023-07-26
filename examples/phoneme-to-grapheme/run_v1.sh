python run_translation.py \
    --model_name_or_path charsiu/g2p_multilingual_byT5_tiny_12_layers_100 \
    --tokenizer_name google/byt5-small \
    --dataset_names bookbot/timit_p2g bookbot/libriphone_p2g bookbot/common-voice-accent-us_p2g bookbot/common-voice-accent-gb_p2g bookbot/common-voice-accent-au_p2g bookbot/common-voice-accent-nz_p2g bookbot/common-voice-accent-ca_p2g bookbot/common-voice-accent-in_p2g bookbot/bookbot_en_v1-v2_v2_p2g \
    --output_dir exp/p2g_charsiu_byt5_tiny_12_layers_100_multi \
    --train_split_names train train train train train train train train train \
    --dataset_probabilities 0.163 0.155 0.054 0.055 0.055 0.039 0.055 0.055 0.369 \
    --eval_split_names None dev validation validation validation validation validation validation validation \
    --source_text_column_name transcript \
    --target_text_column_name text \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 15 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_strategy epoch \
    --max_source_length 512 \
    --max_target_length 512 \
    --val_max_target_length 512 \
    --pad_to_max_length True \
    --overwrite_output_dir \
    --do_train --do_eval \
    --fp16 \
    --optim adamw_torch_fused \
    --predict_with_generate \
    --report_to tensorboard \
    --use_auth_token \
    --push_to_hub \
    --save_total_limit 10 \
    --metric_for_best_model wer \
    --load_best_model_at_end True \
    --hub_private_repo True \
    --hub_model_id bookbot/p2g_charsiu_byt5_tiny_12_layers_100_multi \
    --chars_to_ignore , ? . ! \\- \; \: \" “ % ‘ ” � 
    
    
#   --max_train_samples 100 \
#    --max_eval_samples 100 \