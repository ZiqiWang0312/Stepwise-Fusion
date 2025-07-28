
model_name=FEDformer

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 48 \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 72 \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 96 \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
#   --model $model_name \
#   --data only_do  \
#   --features M \
#   --target shawo\
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 120 \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do.csv  \
#   --model_id BTH_FEDformer_epoch10 \
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
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id 2018_alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 72 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 120 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path alabama_2018.csv  \
  --model_id alabama_FEDformer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target 2423160\
  --seq_len 48 \
  --label_len 24 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \