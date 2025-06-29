if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=PDF

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021

for pred_len in 96 192
do
  CUDA_VISIBLE_DEVICES=0,1 \
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 32 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.35\
      --fc_dropout 0.15 \
      --kernel_list 3 7 9 11 \
      --period 24\
      --patch_len 1\
      --stride 1\
      --des 'Exp'\
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --use_multi_gpu --devices 0,1\
      --itr 1 --batch_size 24 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
  CUDA_VISIBLE_DEVICES=0,1 \
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 32 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.35\
      --fc_dropout 0.15 \
      --kernel_list 3 7 9 11 \
      --period 24\
      --patch_len 1\
      --stride 1\
      --des 'Exp'\
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --use_multi_gpu --devices 0,1\
      --itr 1 --batch_size 24 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
