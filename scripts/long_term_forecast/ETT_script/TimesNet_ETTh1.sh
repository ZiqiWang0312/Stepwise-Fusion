# export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv  \
#   --model_id data_do_96_48_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 24 \
#   --dec_in 24 \
#   --c_out 24 \
#   --d_model 32 \
#   --d_ff 64 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path Beijing_TN_6P.csv  \
#   --model_id data_do_96_48_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target 怀柔水库\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

  # python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./data/data_series/ \
  # --data_path BTH_TN_24P.csv  \
  # --model_id data_do_alabma_TimesNet_epoch10 \
  # --model $model_name \
  # --data only_do  \
  # --features M \
  # --target shawo \
  # --seq_len 48 \
  # --label_len 24 \
  # --pred_len 72 \
  # --e_layers 2 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 5 \
  # --dec_in 5 \
  # --c_out 5 \
  # --d_model 16 \
  # --d_ff 32 \
  # --des 'Exp' \
  # --itr 1 \
  # --top_k 6 

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 72 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6 

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 120 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6

    python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id data_do_alabma_TimesNet_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_48_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data station_do  \
#   --features MS \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

#   python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_192_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data station_do  \
#   --features MS \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

  # python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./data/real_data/ \
  # --data_path data_do_9station_19-23_sg.csv  \
  # --model_id data_do_96_48_MS_TimesNet_epoch10 \
  # --model $model_name \
  # --data station_do  \
  # --features MS \
  # --target huairoushuiku \
  # --seq_len 96 \
  # --label_len 48 \
  # --pred_len 48 \
  # --e_layers 2 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 9 \
  # --dec_in 9 \
  # --c_out 1 \
  # --d_model 16 \
  # --d_ff 32 \
  # --des 'Exp' \
  # --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_96_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features MS \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

#   python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_192_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features MS \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_336_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features MS \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 1 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv    \
#   --model_id data_do_96_test192_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_test336_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9_678.csv    \
#   --model_id data_do_96_96 \
#   --model $model_name \
#   --data station_do \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 256 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9_678.csv    \
#   --model_id data_do_96_96 \
#   --model $model_name \
#   --data station_do \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 256 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9_678.csv    \
#   --model_id data_do_96_96 \
#   --model $model_name \
#   --data station_do \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 256 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6  


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9_678.csv    \
#   --model_id data_do_96_96 \
#   --model $model_name \
#   --data station_do \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 9 \
#   --dec_in 9 \
#   --c_out 9 \
#   --d_model 256 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --top_k 6 
