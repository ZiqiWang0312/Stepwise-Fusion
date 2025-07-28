
model_name=Informer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
  --model $model_name \
  --data only_do  \
  --features M \
  --target shawo\
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path BTH_TN_24P.csv  \
  --model_id BTH_Informer_epoch10 \
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

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1