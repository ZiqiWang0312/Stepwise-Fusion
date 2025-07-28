seq_len=48
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=3
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path beijing_do.csv \
  --model_id beijing_TimeMixer \
  --model TimeMixer \
  --data only_do \
  --features MS \
  --target huairoushuiku \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 10 \
  --e_layers $e_layers \
  --enc_in 6 \
  --c_out 6 \
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
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./data/data_series/ \
  # --data_path beijing_do.csv \
  # --model_id beijing_Nonstationary_Transformer \
  # --model Nonstationary_Transformer \
  # --data only_do  \
  # --features MS \
  # --target huairoushuiku \
  # --seq_len 48 \
  # --label_len 24 \
  # --pred_len 24 \
  # --e_layers 2 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 6 \
  # --dec_in 6 \
  # --c_out 1 \
  # --d_ff 32 \
  # --des 'Exp' \
  # --itr 1 \
  # --p_hidden_dims 256 256 \
  # --p_hidden_layers 2 \
  # --d_model 128

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv \
#   --model_id beijing_Autoformer \
#   --model Autoformer \
#   --data only_do \
#   --features MS \
#   --target huairoushuiku \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --itr 1

#   python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv \
#   --model_id beijing_FEDformer \
#   --model FEDformer \
#   --data only_do \
#   --features MS \
#   --target huairoushuiku \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \

#     python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv \
#   --model_id beijing_LightTS \
#   --model LightTS \
#   --data only_do  \
#   --features MS \
#   --target huairoushuiku \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
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

#   python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv \
#   --model_id beijing_Dlinear \
#   --model DLinear \
#   --data only_do \
#   --features MS \
#   --target huairoushuiku \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 24 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --n_heads 2 \
#   --itr 1