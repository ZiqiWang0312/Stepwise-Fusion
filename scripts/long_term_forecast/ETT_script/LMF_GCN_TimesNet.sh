model_name=LMF_GCN_TimesNet

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./data/data_series/ \
#   --data_path beijing_do_sg_5-9.csv  \
#   --model_id data_do_96_48_MS_TimesNet_epoch10 \
#   --model $model_name \
#   --data station_do  \
#   --features M \
#   --target huairoushuiku\
#   --seq_len 24 \
#   --label_len 12 \
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

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/data_series/ \
  --data_path beijing_do.csv  \
  --model_id data_do_96_48_MS_TimesNet_epoch10 \
  --model $model_name \
  --data station_do  \
  --features M \
  --target huairoushuiku\
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 6 

  # python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./data/data_series/ \
  # --data_path BTH_5-9.csv  \
  # --model_id data_do_96_48_MS_TimesNet_epoch10 \
  # --model $model_name \
  # --data station_do  \
  # --features M \
  # --target shawo\
  # --seq_len 24 \
  # --label_len 12 \
  # --pred_len 96 \
  # --e_layers 2 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 24 \
  # --dec_in 24 \
  # --c_out 24 \
  # --d_model 32 \
  # --d_ff 64 \
  # --des 'Exp' \
  # --itr 1 \
  # --top_k 6 

  #   python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./data/data_series/ \
  # --data_path alabama_18_5-9.csv  \
  # --model_id data_do_96_48_MS_TimesNet_epoch10 \
  # --model $model_name \
  # --data station_do  \
  # --features M \
  # --target 2423160\
  # --seq_len 24 \
  # --label_len 12 \
  # --pred_len 24 \
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