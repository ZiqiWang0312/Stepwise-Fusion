
model_name=Pyraformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/real_data/ \
  --data_path data_do_9station_19-23_sg.csv  \
  --model_id data_do_96_48_MS_Pyraformer_epoch10_final \
  --model $model_name \
  --data only_do  \
  --features MS \
  --target huairoushuiku\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_96_MS_Pyraformer_epoch10_final \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_192_MS_Pyraformer_epoch10_final \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/real_data/ \
#   --data_path data_do_9station_19-23_sg.csv  \
#   --model_id data_do_96_336_MS_Pyraformer_epoch10_final \
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