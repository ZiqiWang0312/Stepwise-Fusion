model_name=TimeMixer

seq_len=48
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=32

#  python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path alabama_4h.csv \
#   --model_id alabama \
#   --model $model_name \
#   --data only_do \
#   --features M \
#   --target 2423160 \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 192 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 5 \
#   --dec_in 5 \
#   --c_out 5 \
#   --des 'Exp' \
#   --n_heads 2 \
#   --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv \
  --model_id BTH_TimeMixer \
  --model $model_name \
  --data only_do \
  --features MS \
  --target shawo \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 24 \
  --e_layers $e_layers \
  --enc_in 24 \
  --c_out 24 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 32 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv \
#   --model_id BTH_TimeMixer \
#   --model $model_name \
#   --data only_do \
#   --features MS \
#   --target shawo \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 48 \
#   --e_layers $e_layers \
#   --enc_in 24 \
#   --c_out 24 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv \
#   --model_id BTH_TimeMixer \
#   --model $model_name \
#   --data only_do \
#   --features MS \
#   --target shawo \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 72 \
#   --e_layers $e_layers \
#   --enc_in 24 \
#   --c_out 24 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv \
#   --model_id BTH_TimeMixer \
#   --model $model_name \
#   --data only_do \
#   --features MS \
#   --target shawo \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 96 \
#   --e_layers $e_layers \
#   --enc_in 24 \
#   --c_out 24 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv \
#   --model_id BTH_TimeMixer \
#   --model $model_name \
#   --data only_do \
#   --features MS \
#   --target shawo \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 120 \
#   --e_layers $e_layers \
#   --enc_in 24 \
#   --c_out 24 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

#   python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path BTH_TN_24P.csv \
#   --model_id BTH_TimeMixer \
#   --model $model_name \
#   --data only_do \
#   --features MS \
#   --target shawo \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --enc_in 24 \
#   --c_out 24 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 32 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window