if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ili" ]; then
    mkdir ./logs/ili
fi

export CUDA_VISIBLE_DEVICES=0

seq_len=36
label_len=18
model_name=MSPredictor

pred_len=24
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --top_k 5 \
    --conv_channel 8 \
    --skip_channel 16 \
    --node_dim 50 \
    --batch_size 16 \
    --itr 1

pred_len=36
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 512 \
    --top_k 5 \
    --conv_channel 16 \
    --skip_channel 32 \
    --node_dim 100 \
    --batch_size 16 \
    --itr 1

pred_len=48
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 512 \
    --top_k 5 \
    --conv_channel 16 \
    --skip_channel 32 \
    --node_dim 100 \
    --batch_size 16 \
    --itr 1

pred_len=60
python -u run_longExp.py \
     --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --freq h \
    --target 'OT' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 512 \
    --top_k 5 \
    --conv_channel 16 \
    --skip_channel 32 \
    --node_dim 100 \
    --batch_size 16 \
    --itr 1