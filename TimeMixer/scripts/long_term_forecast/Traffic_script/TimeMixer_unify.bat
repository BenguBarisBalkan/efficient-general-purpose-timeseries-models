@echo off

call ..\..\..\..\venv\Scripts\activate.bat

rem set CUDA_VISIBLE_DEVICES=0

set model_name=TimeMixer

set seq_len=96
set e_layers=3
set down_sampling_layers=3
set down_sampling_window=2
set learning_rate=0.01
set d_model=32
set d_ff=64
set batch_size=8

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_%seq_len%_96 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 96 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_%seq_len%_192 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 192 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_%seq_len%_336 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 336 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_%seq_len%_720 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 720 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%